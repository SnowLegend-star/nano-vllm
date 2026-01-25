from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        # 1. 初始化 xxh64 哈希对象（核心：高性能非加密哈希）
        h = xxhash.xxh64()
        
        # 2. 若有前缀哈希，先更新前缀（滚动哈希的核心）
        if prefix != -1:
            # prefix转8字节小端序字节串，更新到哈希对象
            h.update(prefix.to_bytes(8, "little"))
        
        # 3. 把token_ids转成字节串，更新哈希对象（核心：计算token_ids的哈希）
        h.update(np.array(token_ids).tobytes())
        
        # 4. 返回哈希值的整数形式（而非字节/十六进制，方便存储和比较）
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0     # 只有引用为0的block才可能被重新分配
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
    # 空闲的block数量大于这个seq需要的block数量才可以分配
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        # 断言：分配前序列的块表必须为空（防止重复分配，保证逻辑安全）
        assert not seq.block_table
        h = -1
        cache_miss = False

        # 遍历该序列需要的所有块（seq.num_blocks是序列的总块数）
        for i in range(seq.num_blocks):
            # 1. 获取第i个块对应的token_ids（seq.block(i)是Sequence类的方法，切分token_ids为块）
            token_ids = seq.block(i)

            # 2. 计算该块的哈希值（仅当token_ids长度等于block_size时计算，否则是不完整块，无法复用）
            # 逻辑：如果块是完整的 → 基于当前哈希h（滚动哈希）计算；否则哈希=-1（标记为不可复用）
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

            # 3. 查哈希表：获取该哈希对应的块ID（无则返回-1）
            block_id = self.hash_to_block_id.get(h, -1)

            # 4. 检查缓存是否命中：
            # 未命中条件：块ID不存在（-1） OR 块的token_ids和当前不一致（哈希碰撞防护）
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True  # 标记为缓存未命中

            # 5. 分配块：未命中则取空闲块，命中则复用
            if cache_miss:
                # 未命中 → 从空闲列表取第一个块ID（FIFO）
                block_id = self.free_block_ids[0]
                # 实际分配块（初始化块的token_ids、哈希、引用计数等）
                block = self._allocate_block(block_id)
            else:
                # 命中 → 累加已缓存token数（该块的token无需重新处理）
                seq.num_cached_tokens += self.block_size
                # 检查该块是否已被其他序列使用
                if block_id in self.used_block_ids:
                    # 已使用 → 复用该块，引用计数+1（表示多一个序列在用）
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 未被使用 → 初始化该块（首次使用）
                    block = self._allocate_block(block_id)

            # 6. 维护哈希表：仅当哈希有效（h≠-1，即完整块）时更新
            if h != -1:
                block.update(h, token_ids)  # 更新块的哈希和token_ids
                self.hash_to_block_id[h] = block_id  # 哈希表映射：哈希→块ID

            # 7. 记录该块ID到序列的块表（序列和块绑定）
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        # 1. 获取序列的块表（记录所有已分配的缓存块ID）
        block_table = seq.block_table
        # 2. 获取最后一个块对象（追加token后，只需要操作最后一个块）
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            # 断言：最后一个块的哈希≠-1（说明上一个块是填满且可复用的，符合逻辑）
            assert last_block.hash != -1
            # 1. 从空闲块列表取第一个块ID（FIFO分配）
            block_id = self.free_block_ids[0]
            # 2. 初始化该新块（设置默认状态：hash=-1，ref_count=1等）
            self._allocate_block(block_id)
            # 3. 将新块ID追加到序列的块表（序列现在多了一个空块）
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            # 断言：最后一个块的哈希=-1（说明该块是未填满的临时块，符合逻辑）
            assert last_block.hash == -1
            # 1. 获取最后一个块的所有token_ids（此时块刚好填满）
            token_ids = seq.block(seq.num_blocks-1)
            # 2. 计算哈希的前缀：若块表长度>1，取倒数第二个块的哈希；否则为-1（滚动哈希）
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            # 3. 计算当前块的哈希（调用之前的compute_hash，滚动哈希）
            h = self.compute_hash(token_ids, prefix)
            # 4. 更新最后一个块的哈希和token_ids（标记为可复用）
            last_block.update(h, token_ids)
            # 5. 维护哈希映射表：哈希值→块ID（供后续allocate时复用）
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
