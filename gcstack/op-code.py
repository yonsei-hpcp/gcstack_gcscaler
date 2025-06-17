#!/usr/bin/python3

# This code counts the number of instruction using an opcode per-kernel - hanna

import os, re, sys, gzip

def check_effective_instruction(line):
    '''
    Check given `line' is effective instruction.
    `line' will be read from `kernel-{}.traceg.gz'.
    '''
    if line == '':
        return False
    if line[0] in ['-', '#']:
        return False
    for keyword in ["thread block", "warp", "insts"]:
        if line.startswith(keyword):
            return False
    return True

def check_effective_kernel(line):
    '''
    Check given `line' is effective file name for kernel.
    `line' will be read from `kernelslist.gz'.
    '''
    return line.startswith("kernel")

def find_file_paths(trace_path, filename="kernelslist.g"):
    '''
    Get all paths of `kernel-{}.traceg.gz' file by read `kernelslist.gz' through walking `trace_path'.
    It will return paths of `kernel-{}.trace.gz' as list type.
    '''
    print("find_file_paths -", trace_path)
    
    rst = list()
    for root, _, files in os.walk(trace_path, followlinks=True):
        if filename not in files:
            continue
        with open (os.path.join(root, filename), 'r') as file:
            data = [i.strip() for i in file.readlines()]
        for kernel_file in data:
            if not check_effective_kernel(kernel_file):
                continue
            kernel_filepath = os.path.join(root, kernel_file)
            assert os.path.isfile(kernel_filepath), "{} is not exist.".format(kernel_filepath)
            rst.append(kernel_filepath)
    return rst

def read_opcode(opcode_path="/data/accel-sim_YA/gpu-simulator/ISA_Def/turing_opcode.h"):
    '''
    Read opcode from header file in ISA_Def directory.
    '''
    print("read_opcode -", opcode_path)
    
    with open(opcode_path, 'r') as file:
        data = file.read()
    opcode_code = re.findall(r'\"(.+)\".+\((.+)\,\ (.+)\)\}', data)
    opcode_dict = {opcode[0]: (opcode[1], opcode[2]) for opcode in opcode_code}
    return opcode_dict

def count(file_path, opocde_dict):
    '''
    Function of conducting actual count opcode.
    '''
    count_dict = {opcode: 0 for opcode in opcode_dict}

    if file_path.split('.')[-1] == "gz":
        with gzip.open(file_path, 'rb') as file:
            trace = file.readlines()
        trace = [line.decode() for line in trace]
    else:
        with open(file_path, 'r') as file:
            trace = file.readlines()
    trace = [i.strip() for i in trace]
    
    for line in trace:
        if not check_effective_instruction(line):
            continue
        line = line.strip().split(' ')
        if len(line) == 1:
            continue
    
        reg_dsts_num = int(line[2])
        opcode = line[3 + reg_dsts_num].split('.')[0]
        count_dict[opcode] += 1

    return file_path, count_dict

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("trace_path is not given.")
        exit(0)
    
    trace_path = sys.argv[1]

    file_paths = find_file_paths(trace_path)
    opcode_dict = read_opcode() 

    # print(file_paths)
    # print(opcode_dict)

   
    from multiprocessing import Pool
    with Pool() as pool:
        starmap_rst = pool.starmap(count, [[file_path, opcode_dict] for file_path in file_paths])

    count_dicts = {file_path: dict() for file_path in file_paths}

    for file_path, count_dict in starmap_rst:
        for opcode in opcode_dict:
            count_dicts[file_path][opcode] = count_dict[opcode]
        # for opcode in count_dict:
        #     count_dicts[opcode] += count_dict[opcode]

    print('-' * 20)
    print("Instruction Opcode Category " + " ".join([file_path.replace(trace_path, '') for file_path in file_paths]))
    for opcode in opcode_dict:
        print(opcode, *opcode_dict[opcode], end='')
        for file_path in file_paths:
            print(" {}".format(count_dicts[file_path][opcode]), end='')
        print()
    
    print('-' * 20)
