/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) cart_pole_expl_vde_forw_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s1[23] = {4, 4, 0, 4, 8, 12, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[3] = {0, 0, 0};

/* cart_pole_expl_vde_forw:(i0[4],i1[4x4],i2[4],i3,i4[])->(o0[4],o1[4x4],o2[4]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a5, a6, a7, a8, a9;
  a0=arg[0]? arg[0][2] : 0;
  if (res[0]!=0) res[0][0]=a0;
  a0=arg[0]? arg[0][3] : 0;
  if (res[0]!=0) res[0][1]=a0;
  a1=arg[3]? arg[3][0] : 0;
  a2=2.0000000000000001e-01;
  a3=arg[0]? arg[0][1] : 0;
  a4=sin(a3);
  a4=(a2*a4);
  a5=5.0000000000000000e-01;
  a6=(a5*a0);
  a7=(a6*a0);
  a8=9.8100000000000005e+00;
  a9=cos(a3);
  a9=(a8*a9);
  a7=(a7+a9);
  a9=(a4*a7);
  a9=(a1+a9);
  a10=2.;
  a11=sin(a3);
  a11=(a2*a11);
  a12=sin(a3);
  a13=(a11*a12);
  a13=(a10+a13);
  a9=(a9/a13);
  if (res[0]!=0) res[0][2]=a9;
  a14=cos(a3);
  a15=(a1*a14);
  a16=1.0000000000000001e-01;
  a17=(a16*a0);
  a18=(a17*a0);
  a19=cos(a3);
  a20=(a18*a19);
  a21=sin(a3);
  a22=(a20*a21);
  a15=(a15+a22);
  a22=2.1582000000000004e+01;
  a23=sin(a3);
  a23=(a22*a23);
  a15=(a15+a23);
  a23=sin(a3);
  a23=(a2*a23);
  a24=sin(a3);
  a25=(a23*a24);
  a10=(a10+a25);
  a10=(a5*a10);
  a15=(a15/a10);
  a25=(-a15);
  if (res[0]!=0) res[0][3]=a25;
  a25=arg[1]? arg[1][2] : 0;
  if (res[1]!=0) res[1][0]=a25;
  a25=arg[1]? arg[1][3] : 0;
  if (res[1]!=0) res[1][1]=a25;
  a26=cos(a3);
  a27=arg[1]? arg[1][1] : 0;
  a28=(a26*a27);
  a28=(a2*a28);
  a28=(a7*a28);
  a29=(a5*a25);
  a29=(a0*a29);
  a30=(a6*a25);
  a29=(a29+a30);
  a30=sin(a3);
  a31=(a30*a27);
  a31=(a8*a31);
  a29=(a29-a31);
  a29=(a4*a29);
  a28=(a28+a29);
  a28=(a28/a13);
  a29=(a9/a13);
  a31=cos(a3);
  a32=(a31*a27);
  a32=(a2*a32);
  a32=(a12*a32);
  a33=cos(a3);
  a34=(a33*a27);
  a34=(a11*a34);
  a32=(a32+a34);
  a32=(a29*a32);
  a28=(a28-a32);
  if (res[1]!=0) res[1][2]=a28;
  a28=(a16*a25);
  a28=(a0*a28);
  a25=(a17*a25);
  a28=(a28+a25);
  a28=(a19*a28);
  a25=sin(a3);
  a32=(a25*a27);
  a32=(a18*a32);
  a28=(a28-a32);
  a28=(a21*a28);
  a32=cos(a3);
  a34=(a32*a27);
  a34=(a20*a34);
  a28=(a28+a34);
  a34=sin(a3);
  a35=(a34*a27);
  a35=(a1*a35);
  a28=(a28-a35);
  a35=cos(a3);
  a36=(a35*a27);
  a36=(a22*a36);
  a28=(a28+a36);
  a28=(a28/a10);
  a36=(a15/a10);
  a37=cos(a3);
  a38=(a37*a27);
  a38=(a2*a38);
  a38=(a24*a38);
  a39=cos(a3);
  a27=(a39*a27);
  a27=(a23*a27);
  a38=(a38+a27);
  a38=(a5*a38);
  a38=(a36*a38);
  a28=(a28-a38);
  a28=(-a28);
  if (res[1]!=0) res[1][3]=a28;
  a28=arg[1]? arg[1][6] : 0;
  if (res[1]!=0) res[1][4]=a28;
  a28=arg[1]? arg[1][7] : 0;
  if (res[1]!=0) res[1][5]=a28;
  a38=arg[1]? arg[1][5] : 0;
  a27=(a26*a38);
  a27=(a2*a27);
  a27=(a7*a27);
  a40=(a5*a28);
  a40=(a0*a40);
  a41=(a6*a28);
  a40=(a40+a41);
  a41=(a30*a38);
  a41=(a8*a41);
  a40=(a40-a41);
  a40=(a4*a40);
  a27=(a27+a40);
  a27=(a27/a13);
  a40=(a31*a38);
  a40=(a2*a40);
  a40=(a12*a40);
  a41=(a33*a38);
  a41=(a11*a41);
  a40=(a40+a41);
  a40=(a29*a40);
  a27=(a27-a40);
  if (res[1]!=0) res[1][6]=a27;
  a27=(a16*a28);
  a27=(a0*a27);
  a28=(a17*a28);
  a27=(a27+a28);
  a27=(a19*a27);
  a28=(a25*a38);
  a28=(a18*a28);
  a27=(a27-a28);
  a27=(a21*a27);
  a28=(a32*a38);
  a28=(a20*a28);
  a27=(a27+a28);
  a28=(a34*a38);
  a28=(a1*a28);
  a27=(a27-a28);
  a28=(a35*a38);
  a28=(a22*a28);
  a27=(a27+a28);
  a27=(a27/a10);
  a28=(a37*a38);
  a28=(a2*a28);
  a28=(a24*a28);
  a38=(a39*a38);
  a38=(a23*a38);
  a28=(a28+a38);
  a28=(a5*a28);
  a28=(a36*a28);
  a27=(a27-a28);
  a27=(-a27);
  if (res[1]!=0) res[1][7]=a27;
  a27=arg[1]? arg[1][10] : 0;
  if (res[1]!=0) res[1][8]=a27;
  a27=arg[1]? arg[1][11] : 0;
  if (res[1]!=0) res[1][9]=a27;
  a28=arg[1]? arg[1][9] : 0;
  a38=(a26*a28);
  a38=(a2*a38);
  a38=(a7*a38);
  a40=(a5*a27);
  a40=(a0*a40);
  a41=(a6*a27);
  a40=(a40+a41);
  a41=(a30*a28);
  a41=(a8*a41);
  a40=(a40-a41);
  a40=(a4*a40);
  a38=(a38+a40);
  a38=(a38/a13);
  a40=(a31*a28);
  a40=(a2*a40);
  a40=(a12*a40);
  a41=(a33*a28);
  a41=(a11*a41);
  a40=(a40+a41);
  a40=(a29*a40);
  a38=(a38-a40);
  if (res[1]!=0) res[1][10]=a38;
  a38=(a16*a27);
  a38=(a0*a38);
  a27=(a17*a27);
  a38=(a38+a27);
  a38=(a19*a38);
  a27=(a25*a28);
  a27=(a18*a27);
  a38=(a38-a27);
  a38=(a21*a38);
  a27=(a32*a28);
  a27=(a20*a27);
  a38=(a38+a27);
  a27=(a34*a28);
  a27=(a1*a27);
  a38=(a38-a27);
  a27=(a35*a28);
  a27=(a22*a27);
  a38=(a38+a27);
  a38=(a38/a10);
  a27=(a37*a28);
  a27=(a2*a27);
  a27=(a24*a27);
  a28=(a39*a28);
  a28=(a23*a28);
  a27=(a27+a28);
  a27=(a5*a27);
  a27=(a36*a27);
  a38=(a38-a27);
  a38=(-a38);
  if (res[1]!=0) res[1][11]=a38;
  a38=arg[1]? arg[1][14] : 0;
  if (res[1]!=0) res[1][12]=a38;
  a38=arg[1]? arg[1][15] : 0;
  if (res[1]!=0) res[1][13]=a38;
  a27=arg[1]? arg[1][13] : 0;
  a26=(a26*a27);
  a26=(a2*a26);
  a26=(a7*a26);
  a28=(a5*a38);
  a28=(a0*a28);
  a40=(a6*a38);
  a28=(a28+a40);
  a30=(a30*a27);
  a30=(a8*a30);
  a28=(a28-a30);
  a28=(a4*a28);
  a26=(a26+a28);
  a26=(a26/a13);
  a31=(a31*a27);
  a31=(a2*a31);
  a31=(a12*a31);
  a33=(a33*a27);
  a33=(a11*a33);
  a31=(a31+a33);
  a29=(a29*a31);
  a26=(a26-a29);
  if (res[1]!=0) res[1][14]=a26;
  a26=(a16*a38);
  a26=(a0*a26);
  a38=(a17*a38);
  a26=(a26+a38);
  a26=(a19*a26);
  a25=(a25*a27);
  a25=(a18*a25);
  a26=(a26-a25);
  a26=(a21*a26);
  a32=(a32*a27);
  a32=(a20*a32);
  a26=(a26+a32);
  a34=(a34*a27);
  a34=(a1*a34);
  a26=(a26-a34);
  a35=(a35*a27);
  a35=(a22*a35);
  a26=(a26+a35);
  a26=(a26/a10);
  a37=(a37*a27);
  a37=(a2*a37);
  a37=(a24*a37);
  a39=(a39*a27);
  a39=(a23*a39);
  a37=(a37+a39);
  a37=(a5*a37);
  a36=(a36*a37);
  a26=(a26-a36);
  a26=(-a26);
  if (res[1]!=0) res[1][15]=a26;
  a26=arg[2]? arg[2][2] : 0;
  if (res[2]!=0) res[2][0]=a26;
  a26=arg[2]? arg[2][3] : 0;
  if (res[2]!=0) res[2][1]=a26;
  a36=(1./a13);
  a37=cos(a3);
  a39=arg[2]? arg[2][1] : 0;
  a37=(a37*a39);
  a37=(a2*a37);
  a7=(a7*a37);
  a37=(a5*a26);
  a37=(a0*a37);
  a6=(a6*a26);
  a37=(a37+a6);
  a6=sin(a3);
  a6=(a6*a39);
  a8=(a8*a6);
  a37=(a37-a8);
  a4=(a4*a37);
  a7=(a7+a4);
  a7=(a7/a13);
  a9=(a9/a13);
  a13=cos(a3);
  a13=(a13*a39);
  a13=(a2*a13);
  a12=(a12*a13);
  a13=cos(a3);
  a13=(a13*a39);
  a11=(a11*a13);
  a12=(a12+a11);
  a9=(a9*a12);
  a7=(a7-a9);
  a36=(a36+a7);
  if (res[2]!=0) res[2][2]=a36;
  a14=(a14/a10);
  a16=(a16*a26);
  a0=(a0*a16);
  a17=(a17*a26);
  a0=(a0+a17);
  a19=(a19*a0);
  a0=sin(a3);
  a0=(a0*a39);
  a18=(a18*a0);
  a19=(a19-a18);
  a21=(a21*a19);
  a19=cos(a3);
  a19=(a19*a39);
  a20=(a20*a19);
  a21=(a21+a20);
  a20=sin(a3);
  a20=(a20*a39);
  a1=(a1*a20);
  a21=(a21-a1);
  a1=cos(a3);
  a1=(a1*a39);
  a22=(a22*a1);
  a21=(a21+a22);
  a21=(a21/a10);
  a15=(a15/a10);
  a10=cos(a3);
  a10=(a10*a39);
  a2=(a2*a10);
  a24=(a24*a2);
  a3=cos(a3);
  a3=(a3*a39);
  a23=(a23*a3);
  a24=(a24+a23);
  a5=(a5*a24);
  a15=(a15*a5);
  a21=(a21-a15);
  a14=(a14+a21);
  a14=(-a14);
  if (res[2]!=0) res[2][3]=a14;
  return 0;
}

CASADI_SYMBOL_EXPORT int cart_pole_expl_vde_forw(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int cart_pole_expl_vde_forw_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int cart_pole_expl_vde_forw_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void cart_pole_expl_vde_forw_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int cart_pole_expl_vde_forw_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void cart_pole_expl_vde_forw_release(int mem) {
}

CASADI_SYMBOL_EXPORT void cart_pole_expl_vde_forw_incref(void) {
}

CASADI_SYMBOL_EXPORT void cart_pole_expl_vde_forw_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int cart_pole_expl_vde_forw_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int cart_pole_expl_vde_forw_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real cart_pole_expl_vde_forw_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* cart_pole_expl_vde_forw_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* cart_pole_expl_vde_forw_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* cart_pole_expl_vde_forw_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    case 3: return casadi_s2;
    case 4: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* cart_pole_expl_vde_forw_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int cart_pole_expl_vde_forw_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
