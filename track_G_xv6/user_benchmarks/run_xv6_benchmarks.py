#!/usr/bin/env python3
import sys
import os
import json
import re
import pexpect

def run_benchmarks(cpus=1, timeout=60):
    cmd = f"make qemu CPUS={cpus}"
    print(f"[*] Spawning QEMU with CPUS={cpus}...")
    child = pexpect.spawn(cmd, encoding='utf-8', timeout=timeout)
    
    # Wait for xv6 shell prompt
    child.expect_exact('$ ')
    print("[+] xv6 booted successfully!")

    # 1. Run forktest to verify kernel correctness
    print("[*] Running forktest in xv6...")
    child.sendline("forktest")
    child.expect_exact('$ ', timeout=60)
    fork_output = child.before
    print("[+] forktest output:")
    print(fork_output)

    # 2. Run schedtest
    print("[*] Running schedtest in xv6...")
    child.sendline("schedtest")
    child.expect_exact('$ ', timeout=90)
    sched_output = child.before
    print("[+] schedtest output:")
    print(sched_output)

    # 3. Run bcachetest
    print("[*] Running bcachetest in xv6...")
    child.sendline("bcachetest")
    child.expect_exact('$ ', timeout=60)
    bcache_output = child.before
    print("[+] bcachetest output:")
    print(bcache_output)

    # Clean shutdown of QEMU using Ctrl-A X
    child.send('\x01x')
    child.expect(pexpect.EOF)
    print("[+] QEMU exited cleanly.")

    # Save raw outputs to JSON
    results = {
        "cpus": cpus,
        "fork_output": fork_output,
        "sched_output": sched_output,
        "bcache_output": bcache_output,
    }
    
    out_dir = "/mnt/c/dev/ot-8/track_G_xv6/results"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "raw_xv6_run.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Saved raw results to {os.path.join(out_dir, 'raw_xv6_run.json')}")

    return results

if __name__ == '__main__':
    cpus = 1
    if len(sys.argv) > 1:
        cpus = int(sys.argv[1])
    run_benchmarks(cpus=cpus)
