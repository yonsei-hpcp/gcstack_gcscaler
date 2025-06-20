// developed by Mahmoud Khairy, Purdue Univ
// abdallm@purdue.edu

#ifndef TRACE_OPCODE_H
#define TRACE_OPCODE_H

#include <string>
#include <unordered_map>
#include "abstract_hardware_model.h"

enum TraceInstrOpcode {
  // Volta (includes common insts for others cards as well)
  OP_FADD = 1,
  OP_FADD32I,
  OP_FCHK,
  OP_FFMA32I,
  OP_FFMA,
  OP_FMNMX,
  OP_FMUL,
  OP_FMUL32I,
  OP_FSEL,
  OP_FSET,
  OP_FSETP,
  OP_FSWZADD,
  OP_MUFU,
  OP_HADD2,
  OP_HADD2_32I,
  OP_HFMA2,
  OP_HFMA2_32I,
  OP_HMUL2,
  OP_HMUL2_32I,
  OP_HSET2,
  OP_HSETP2,
  OP_HMMA,
  OP_DADD,
  OP_DFMA,
  OP_DMUL,
  OP_DSETP,
  OP_BMSK,
  OP_BREV,
  OP_FLO,
  OP_IABS,
  OP_IADD,
  OP_IADD3,
  OP_IADD32I,
  OP_IDP,
  OP_IDP4A,
  OP_IMAD,
  OP_IMMA,
  OP_IMNMX,
  OP_IMUL,
  OP_IMUL32I,
  OP_ISCADD,
  OP_ISCADD32I,
  OP_ISETP,
  OP_LEA,
  OP_LOP,
  OP_LOP3,
  OP_LOP32I,
  OP_POPC,
  OP_SHF,
  OP_SHR,
  OP_VABSDIFF,
  OP_VABSDIFF4,
  OP_F2F,
  OP_F2I,
  OP_I2F,
  OP_I2I,
  OP_I2IP,
  OP_FRND,
  OP_MOV,
  OP_MOV32I,
  OP_PRMT,
  OP_SEL,
  OP_SGXT,
  OP_SHFL,
  OP_PLOP3,
  OP_PSETP,
  OP_P2R,
  OP_R2P,
  OP_LD,
  OP_LDC,
  OP_LDG,
  OP_LDL,
  OP_LDS,
  OP_ST,
  OP_STG,
  OP_STL,
  OP_STS,
  OP_MATCH,
  OP_QSPC,
  OP_ATOM,
  OP_ATOMS,
  OP_ATOMG,
  OP_RED,
  OP_CCTL,
  OP_CCTLL,
  OP_ERRBAR,
  OP_MEMBAR,
  OP_CCTLT,
  OP_TEX,
  OP_TLD,
  OP_TLD4,
  OP_TMML,
  OP_TXD,
  OP_TXQ,
  OP_BMOV,
  OP_BPT,
  OP_BRA,
  OP_BREAK,
  OP_BRX,
  OP_BSSY,
  OP_BSYNC,
  OP_CALL,
  OP_EXIT,
  OP_JMP,
  OP_JMX,
  OP_KILL,
  OP_NANOSLEEP,
  OP_RET,
  OP_RPCMOV,
  OP_RTT,
  OP_WARPSYNC,
  OP_YIELD,
  OP_B2R,
  OP_BAR,
  OP_CS2R,
  OP_CSMTEST,
  OP_DEPBAR,
  OP_GETLMEMBASE,
  OP_LEPC,
  OP_NOP,
  OP_PMTRIG,
  OP_R2B,
  OP_S2R,
  OP_SETCTAID,
  OP_SETLMEMBASE,
  OP_VOTE,
  OP_VOTE_VTG,
  // unique insts for pascal
  OP_RRO,
  OP_DMNMX,
  OP_DSET,
  OP_BFE,
  OP_BFI,
  OP_ICMP,
  OP_IMADSP,
  OP_SHL,
  OP_XMAD,
  OP_CSET,
  OP_CSETP,
  OP_TEXS,
  OP_TLD4S,
  OP_TLDS,
  OP_CAL,
  OP_JCAL,
  OP_PRET,
  OP_BRK,
  OP_PBK,
  OP_CONT,
  OP_PCNT,
  OP_PEXIT,
  OP_SSY,
  OP_SYNC,
  OP_PSET,
  OP_VMNMX,
  OP_ISET,
  // unique insts for turing
  OP_BMMA,
  OP_MOVM,
  OP_LDSM,
  OP_R2UR,
  OP_S2UR,
  OP_UBMSK,
  OP_UBREV,
  OP_UCLEA,
  OP_UFLO,
  OP_UIADD3,
  OP_UIMAD,
  OP_UISETP,
  OP_ULDC,
  OP_ULEA,
  OP_ULOP,
  OP_ULOP3,
  OP_ULOP32I,
  OP_UMOV,
  OP_UP2UR,
  OP_UPLOP3,
  OP_UPOPC,
  OP_UPRMT,
  OP_UPSETP,
  OP_UR2UP,
  OP_USEL,
  OP_USGXT,
  OP_USHF,
  OP_USHL,
  OP_USHR,
  OP_VOTEU,
  OP_SUATOM,
  OP_SULD,
  OP_SURED,
  OP_SUST,
  OP_BRXU,
  OP_JMXU,
  // unique insts for kepler
  OP_FCMP,
  OP_FSWZ,
  OP_ISAD,
  OP_LDSLK,
  OP_STSCUL,
  OP_SUCLAMP,
  OP_SUBFM,
  OP_SUEAU,
  OP_SULDGA,
  OP_SUSTGA,
  OP_ISUB,
  // unique insts for ampere
  OP_HMNMX2,
  OP_DMMA,
  OP_I2FP,
  OP_F2IP,
  OP_LDGDEPBAR,
  OP_LDGSTS,
  OP_REDUX,
  OP_UF2FP,
  OP_F2FP, // - YA
  OP_SUQUERY,
  SASS_NUM_OPCODES /* The total number of opcodes. */
};
typedef enum TraceInstrOpcode sass_op_type;

struct OpcodeChar {
  OpcodeChar(unsigned m_opcode, unsigned m_opcode_category) {
    opcode = m_opcode;
    opcode_category = m_opcode_category;
  }
  unsigned opcode;
  unsigned opcode_category;
};

#endif
