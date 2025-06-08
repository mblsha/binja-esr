# Hardware-in-the-Loop Test Framework for SC62015

This document outlines a complete framework for validating the SC62015 emulator against real hardware. The core principle is to use the real CPU as the "source of truth" and systematically compare its behavior against the emulator for every instruction.

## 1. High-Level Workflow

The process is managed by a Python script on a PC (the "Orchestrator") which communicates with a BASIC program on the Sharp Wizard (the "Harness").

## 2. Communication Protocol Specification

The protocol uses ASCII commands and data over a standard serial connection.

*   **Connection:** 9600 baud, 8 data bits, No parity, 1 stop bit (9600,N,8,1).
*   **Termination:** All commands and data sent from the PC are terminated by a newline (`\n`). All responses from the device are terminated by a newline.

### **Command: `P` (Ping)**

*   **Purpose:** Verify that the harness is running and the connection is alive.
*   **PC Sends:**
    ```
    P\n
    ```
*   **Device Responds:**
    ```
    PONG\n
    OK\n
    ```
*   **Device On-Screen Display:** `CMD: PING`, then `PONG`.

### **Command: `L` (Load Code)**

*   **Purpose:** Loads a block of machine code into the device's RAM.
*   **PC Sends:**
    1.  `L\n`
    2.  `&H<address_hex>\n` (e.g., `&H8000\n`)
    3.  `<length_dec>\n` (e.g., `24\n`)
    4.  `<byte1_hex>\n` (e.g., `08\n`)
    5.  `<byte2_hex>\n` (e.g., `42\n`)
    6.  ... (repeats for `<length_dec>` bytes)
*   **Device Responds:**
    ```
    OK\n
    ```
*   **Device On-Screen Display:** `CMD: LOAD`, `ADDR: <addr> LEN: <len>`, followed by one dot per byte received.

### **Command: `X` (eXecute)**

*   **Purpose:** Executes the loaded machine code via `CALL`.
*   **PC Sends:**
    1.  `X\n`
    2.  `&H<address_hex>\n` (e.g., `&H8000\n`)
*   **Device Responds:**
    ```
    OK\n
    ```
*   **Device On-Screen Display:** `CMD: EXECUTE`, `CALLING <addr>...`, then `EXEC OK` after the `CALL` returns.

### **Command: `R` (Read Memory)**

*   **Purpose:** Reads a block of memory and sends its contents to the PC.
*   **PC Sends:**
    1.  `R\n`
    2.  `&H<address_hex>\n` (e.g., `&H9000\n`)
    3.  `<length_dec>\n` (e.g., `12\n`)
*   **Device Responds:**
    1.  `<byte1_hex>\n` (e.g., `05\n`)
    2.  `<byte2_hex>\n` (e.g., `00\n`)
    3.  ... (repeats for `<length_dec>` bytes)
    4.  `OK\n`
*   **Device On-Screen Display:** `CMD: READ`, `ADDR: <addr> LEN: <len>`, followed by a mini hex dump of the data.

