"""Show what text_processor.py does to a small sample."""

from aqz.preprocessing.text_processor import chunk_text, preprocess_text


sample_text = """Chapter 1: CPU Architecture and Instruction Execution

1.1 Introduction to the Central Processing Unit

The Central Processing Unit (CPU) is the primary computational component of a computer system responsible for executing instructions, performing arithmetic and logical operations, and coordinating the activities of other hardware components. A CPU consists of several functional units, including the Arithmetic Logic Unit (ALU), control unit, registers, and internal communication pathways. These components work together to execute machine instructions stored in memory. The performance of a processor depends on several factors, such as clock frequency, instruction set architecture, pipeline design, memory latency, and the number of instructions executed by a program.

The processor operates through a repeated sequence known as the instruction cycle. During this cycle, the CPU fetches an instruction from memory, decodes it to determine the required operation, obtains the necessary operands, executes the operation, and stores the result. Depending on the instruction, additional stages may be required, such as accessing memory or updating the processor's status flags. Modern processors use sophisticated mechanisms to overlap these stages and execute multiple instructions efficiently.

1.2 Registers and Their Functions

Registers are small, high-speed storage locations located inside the processor. They provide temporary storage for operands, addresses, instructions, and intermediate results. Since registers are directly accessible by the processor's execution units, accessing a register is significantly faster than accessing main memory.

The Program Counter (PC) stores the address of the next instruction to be fetched. After an instruction is fetched, the PC is normally updated to point to the subsequent instruction. However, branch and jump instructions may modify the PC to redirect program execution to a different location. The Instruction Register (IR) stores the instruction currently being decoded or executed. The Memory Address Register (MAR) holds the address of the memory location involved in an operation, while the Memory Data Register (MDR) temporarily stores data being transferred between the processor and memory.

General-purpose registers are used to store operands and intermediate results during program execution. Their availability reduces the number of memory accesses required by a program. Some processors also contain specialized registers such as status registers, which maintain condition flags indicating whether an arithmetic operation produced a zero result, overflow, carry, or negative value.

1.3 Instruction Execution Cycle

Instruction execution begins when the processor retrieves an instruction from the memory location specified by the Program Counter. The instruction is transferred into the Instruction Register, and the PC is updated to point to the next instruction. The control unit examines the opcode and determines the operation to be performed.

During the decode stage, the processor identifies the instruction type, required operands, and addressing mode. If operands are stored in registers, they are read from the register file. If an operand is located in memory, the processor may need to calculate the effective address and perform a memory access.

The execution stage performs the required operation. Arithmetic instructions are handled by the ALU, logical instructions perform operations such as AND, OR, or XOR, and control-transfer instructions modify the flow of execution. Following execution, the result may be written into a destination register or memory location.

The instruction cycle is repeated continuously until the processor encounters an instruction that terminates the program or transfers control to another execution context. The actual implementation of this cycle differs between architectures, especially in processors that use pipelining, superscalar execution, or out-of-order execution.

Chapter 2: Instruction Pipelining

2.1 Concept of Pipelining

Instruction pipelining is a technique used to improve processor throughput by overlapping the execution of multiple instructions. Instead of completing one instruction entirely before beginning the next, the processor divides instruction execution into smaller stages. Different instructions can occupy different stages simultaneously.

A typical five-stage pipeline consists of Instruction Fetch (IF), Instruction Decode (ID), Execute (EX), Memory Access (MEM), and Write Back (WB). During instruction fetch, the processor retrieves an instruction from memory. During decode, the instruction is interpreted and the required registers are read. The execute stage performs arithmetic operations or calculates memory addresses. The memory stage accesses data memory when required, and the write-back stage stores the final result in a register.

In an ideal pipeline, each stage takes one clock cycle and all stages operate concurrently. Once the pipeline is full, ideally one instruction completes during every clock cycle. This improves throughput but does not necessarily reduce the latency of an individual instruction. An instruction may still require five clock cycles to pass through a five-stage pipeline.

2.2 Pipeline Speedup

Consider a processor with five pipeline stages, each requiring one clock cycle. Without pipelining, an instruction would require five clock cycles, and four instructions would require twenty clock cycles if executed sequentially. With pipelining, the first instruction completes after five cycles, while the remaining instructions complete in subsequent cycles. Therefore, four instructions can complete in eight cycles under ideal conditions.

The ideal speedup of a pipeline with k stages approaches k when the number of instructions is sufficiently large and all stages operate efficiently. However, practical processors rarely achieve this theoretical maximum because pipeline stalls, branch instructions, unequal stage delays, and resource conflicts introduce additional cycles.

Pipelining primarily improves instruction throughput rather than reducing the execution latency of a single instruction. A processor with a deeper pipeline may achieve a higher clock frequency because each stage performs less work, but the deeper pipeline may also increase the penalty associated with branch misprediction and pipeline hazards.

2.3 Pipeline Hazards

A pipeline hazard occurs when an instruction cannot proceed during its scheduled clock cycle because of a dependency or resource conflict. Hazards are classified into structural hazards, data hazards, and control hazards.

A structural hazard occurs when multiple instructions require the same hardware resource simultaneously, but the processor does not have enough resources to satisfy both requests. For example, if instruction fetching and data memory access share a single memory unit, an instruction fetch may be delayed when another instruction accesses memory.

A data hazard occurs when an instruction depends on the result of an earlier instruction that has not yet completed. Read After Write (RAW) is the most common data dependency in simple pipelines. For example, if one instruction calculates the value of register R1 and the next instruction immediately uses R1, the second instruction may need to wait until the first instruction produces the result.

Forwarding, also called bypassing, reduces data-hazard delays by transferring results directly from an intermediate pipeline stage to another stage without waiting for the result to be written into the register file. When forwarding cannot resolve a dependency, the processor inserts stalls, often represented by bubbles in the pipeline.

A control hazard occurs when the processor encounters a branch or jump instruction and does not yet know which instruction should execute next. If the processor fetches instructions along the wrong path, those instructions may need to be discarded. Branch prediction attempts to predict the outcome of the branch and reduce the delay caused by control hazards.

Chapter 3: Memory Hierarchy and Cache Organization

3.1 Memory Hierarchy

Computer systems use multiple levels of memory because no single memory technology can simultaneously provide high capacity, low cost, and extremely low access latency. The memory hierarchy organizes storage into levels based on speed, capacity, and cost per bit.

At the top of the hierarchy are CPU registers, which provide the fastest access but have limited capacity. Cache memory is located close to the processor and stores frequently used instructions and data. Main memory provides greater storage capacity but has higher access latency. Secondary storage, such as SSDs and HDDs, provides persistent storage and much larger capacity but is considerably slower than main memory.

The memory hierarchy exploits the principle of locality of reference. Temporal locality states that recently accessed data is likely to be accessed again soon. Spatial locality states that memory locations near a recently accessed location are likely to be accessed in the near future. Programs frequently exhibit both forms of locality, particularly when executing loops or traversing arrays.

3.2 Cache Memory

Cache memory is a small, fast memory that stores copies of frequently accessed blocks from main memory. It reduces the average time required by the processor to access instructions and data. Cache memory is commonly implemented using Static Random Access Memory (SRAM), which is faster than DRAM but more expensive per bit.

Cache memory is organized into cache lines. Each cache line contains a data block and associated metadata. Metadata commonly includes a tag, valid bit, and dirty bit. The tag identifies the memory block stored in the cache line, the valid bit indicates whether the line contains meaningful data, and the dirty bit indicates whether the cached data has been modified.

When the processor requests a memory address, the cache controller checks whether the corresponding memory block is present. If the block is present, a cache hit occurs. If the block is absent, a cache miss occurs, and the block must be obtained from a lower level of the memory hierarchy.

Cache performance is commonly evaluated using the hit ratio, miss ratio, hit time, and miss penalty. The Average Memory Access Time (AMAT) can be calculated as:

AMAT = Hit Time + Miss Rate × Miss Penalty

A higher hit ratio generally improves performance because fewer accesses require the slower memory system. However, a cache with a high hit ratio may still perform poorly if its miss penalty is extremely large.

3.3 Cache Mapping Techniques

Direct mapping allows each memory block to occupy exactly one cache line. The cache line is determined using the memory block number modulo the number of cache lines. Direct mapping is simple and inexpensive, but blocks that map to the same cache line may repeatedly replace each other, causing conflict misses.

Fully associative mapping allows a memory block to be stored in any cache line. This reduces conflict misses because there are no restrictions on placement. However, the cache controller must compare the requested tag with the tags of many cache lines, increasing hardware complexity.

Set-associative mapping combines both approaches. The cache is divided into sets, and each memory block maps to a specific set but may occupy any line within that set. A four-way set-associative cache, for example, contains four cache lines per set. Set-associative mapping offers a balance between hardware complexity and cache utilization.

When a cache set becomes full, a replacement policy selects a line for eviction. Least Recently Used (LRU) replaces the line that has not been accessed for the longest time. First-In First-Out (FIFO) replaces the line that entered the cache earliest. Random replacement selects a line randomly. The effectiveness of a replacement policy depends on the workload and the patterns of memory access.

Chapter 4: Virtual Memory and Address Translation

4.1 Virtual Memory

Virtual memory is a memory management technique that allows processes to use an address space larger than the available physical memory. It provides the illusion that each process has access to a large, continuous memory region, even though the actual data may be distributed across physical memory and secondary storage.

Virtual memory also provides isolation between processes. Each process generally operates within its own virtual address space, preventing one process from directly accessing another process's memory without appropriate permissions. The operating system manages virtual-to-physical address translation and controls which pages are resident in physical memory.

The virtual address space is divided into pages, while physical memory is divided into frames. Pages and frames have equal sizes, allowing a virtual page to be placed into any available physical frame. This flexibility eliminates the requirement that a process occupy one continuous region of physical memory.

4.2 Page Tables and Translation Lookaside Buffer

A page table stores mappings between virtual page numbers and physical frame numbers. A virtual address contains a page number and a page offset. The page number is translated into a physical frame number, while the offset remains unchanged.

Each page table entry may contain a valid bit, protection information, an accessed bit, and a dirty bit. The valid bit indicates whether the mapping is currently available. Protection bits control permitted operations, such as reading, writing, or executing the page.

A Translation Lookaside Buffer (TLB) is a small associative memory that caches recently used address translations. When a virtual address is generated, the processor checks the TLB before accessing the page table. A TLB hit provides the physical frame number quickly, reducing the overhead of address translation.

A TLB miss occurs when the required translation is not present in the TLB. The processor or operating system must retrieve the translation from the page table. A TLB miss does not necessarily imply that the page is absent from physical memory. If the page is resident, the translation can be obtained and execution can continue without loading data from secondary storage.

4.3 Page Faults

A page fault occurs when a process accesses a virtual page that is not currently available in physical memory or when the access requires operating system intervention. The operating system first checks whether the access is valid. If the address is illegal, the process may be terminated. If the access is valid but the page is absent from memory, the operating system locates the page on secondary storage.

The operating system selects a physical frame to hold the required page. If no free frame exists, a page replacement algorithm determines which existing page should be removed. If the selected page has been modified, it may need to be written back to storage before replacement. The required page is then loaded into the chosen frame, and the page table is updated.

After the page becomes available, the processor restarts the instruction that caused the page fault. Page faults are expensive because they may require storage access, operating system execution, and additional memory management operations. Their frequency can significantly affect overall system performance.

Chapter 5: Input/Output Organization

5.1 I/O Communication

Input/output organization enables the processor to communicate with external devices such as keyboards, printers, storage devices, sensors, displays, and network interfaces. These devices operate at different speeds and may use different communication protocols. The I/O system manages data transfers, device status, control signals, and error handling.

Programmed I/O, interrupt-driven I/O, and Direct Memory Access (DMA) are three common approaches to handling data transfers. They differ primarily in the degree of CPU involvement during the transfer.

5.2 Programmed and Interrupt-Driven I/O

In programmed I/O, the CPU directly controls the transfer by executing instructions that check the device status and move data. Polling is frequently used, in which the processor repeatedly checks whether the device is ready. Although this method is straightforward, it wastes CPU time when the device is slow.

Interrupt-driven I/O allows the processor to initiate an operation and perform other tasks while the device works. When the device requires attention, it generates an interrupt. The processor temporarily suspends its current execution, runs an interrupt service routine, handles the device event, and then resumes the interrupted program.

Interrupt-driven I/O improves processor utilization, but it introduces overhead associated with interrupt handling. Excessively frequent interrupts may reduce performance because the processor spends more time responding to device events and less time executing useful instructions.

5.3 Direct Memory Access

Direct Memory Access (DMA) allows data to be transferred between an I/O device and main memory without requiring the CPU to handle each individual data item. The processor initially configures the DMA controller by specifying the memory address, transfer direction, and number of bytes to be transferred.

Once the transfer begins, the DMA controller manages the movement of data. The CPU can continue executing other instructions while the transfer is in progress. After the operation completes, the DMA controller typically generates an interrupt to inform the processor.

DMA is useful for large data transfers involving high-speed devices because it reduces CPU overhead. However, DMA and the CPU may compete for access to the memory bus. Burst-mode DMA transfers a sequence of data items while holding the bus for a period, whereas cycle stealing allows the DMA controller to acquire individual bus cycles between CPU accesses.

Chapter 6: Performance Evaluation

6.1 CPU Performance

CPU performance is influenced by instruction count, clock rate, and the average number of clock cycles required per instruction. CPU execution time can be expressed as:

CPU Time = Instruction Count × CPI × Clock Cycle Time

Since clock cycle time is the reciprocal of clock rate, the equation can also be expressed as:

CPU Time = (Instruction Count × CPI) / Clock Rate

Instruction Count depends on the program and instruction set. CPI, or Cycles Per Instruction, depends on processor implementation and instruction characteristics. Clock rate determines the number of cycles the processor executes per second.

Increasing clock frequency does not always result in proportionally greater performance. A processor with a higher clock rate may execute more cycles per instruction or may spend more time waiting for memory. Performance comparisons should therefore consider the complete execution time rather than clock frequency alone.

6.2 Amdahl's Law

Amdahl's Law describes the limitations of improving only a portion of a computer system or program. If a fraction of execution time is improved while the remaining portion stays unchanged, the total speedup is limited by the unoptimized portion.

The speedup formula is:

Speedup = 1 / [(1 − f) + (f / S)]

Here, f represents the fraction of execution time affected by the improvement, while S represents the speedup of that portion.

For example, if 70 percent of a program can be accelerated by a factor of 4, the overall speedup is:

Speedup = 1 / [0.30 + (0.70 / 4)]

Speedup = 1 / 0.475 ≈ 2.105

Although the optimized portion becomes four times faster, the entire program improves by only approximately 2.1 times because the remaining 30 percent of execution time is unchanged.

Amdahl's Law demonstrates that optimizing a component with a small contribution to total execution time provides limited benefit. Effective performance optimization requires identifying the portions of a system that consume the most time and improving them without introducing excessive overhead elsewhere."""


def main():
    cleaned_text = preprocess_text(sample_text)
    chunks = chunk_text(cleaned_text, max_words=12, overlap=5)

    print("=== ORIGINAL TEXT ===")
    print(sample_text)

    print("=== CLEANED TEXT ===")
    print(cleaned_text)

    print(f"=== CHUNKS ({len(chunks)} total) ===")
    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ({len(chunk.split())} words) ---")
        print(chunk)


if __name__ == "__main__":
    main()
