"""Fixed native ABI around the pinned upstream AOTriton forward kernel."""

import triton
import triton.language as tl
from fwd_kernel import attn_fwd

@triton.jit(do_not_specialize=["Q_LEN", "KV_LEN"])
def canvas(Q,K,V,L,Out, Q_LEN, KV_LEN, D:tl.constexpr, HK:tl.constexpr):
    attn_fwd(Q=Q,K=K,V=V,B=None,A=None,Sm_scale=1.,L=L,Out=Out,
        Q_descale=False,K_descale=False,P_scale=False,P_descale=False,V_descale=False,
        stride_qz=16*Q_LEN*D,stride_qh=Q_LEN*D,stride_qm=D,stride_qk=1,
        stride_kz=HK*KV_LEN*D,stride_kh=KV_LEN*D,stride_kn=D,stride_kk=1,
        stride_vz=HK*KV_LEN*D,stride_vh=KV_LEN*D,stride_vk=D,stride_vn=1,
        stride_oz=16*Q_LEN*D,stride_oh=Q_LEN*D,stride_om=D,stride_on=1,
        stride_bz=0,stride_bh=0,stride_bm=0,stride_bn=0,stride_az=0,stride_ah=0,
        Num_head_q=16,Num_head_k=HK,Num_seqlens=0,cu_seqlens_q=None,cu_seqlens_k=None,
        Max_seqlen_q=Q_LEN,Max_seqlen_k=KV_LEN,seq_strides_q=None,seq_strides_k=None,
        BLOCK_DMODEL=D,Hdim_qk=D,Hdim_vo=D,PADDED_HEAD=False,ENABLE_DROPOUT=False,
        dropout_p=0.,philox_seed_ptr=None,philox_offset1=None,philox_offset2=0,
        philox_seed_output=None,philox_offset_output=None,RETURN_ENCODED_SOFTMAX=False,
        encoded_softmax=None,CAUSAL_TYPE=0,Window_left=-1,Window_right=-1,BIAS_TYPE=0,
        USE_ALIBI=False,INT8=False,INT8_KV=False,USE_P_SCALE=False,PERSISTENT_TYPE=0,
        persistent_atomic_counter=None,Num_CU=40,GRID_CU_MULTIP=1,Batch=1,
        BLOCK_M=32,BLOCK_N=32,PRE_LOAD_V=False,NUM_XCDS=1)
