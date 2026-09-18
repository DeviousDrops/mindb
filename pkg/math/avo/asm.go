//go:build ignore

// Generates the AVX2 asymmetric int8 dot-product kernel: query stays
// float32, codes are int8, converted lane-by-lane via VPMOVSXBD+VCVTDQ2PS
// before the FMA. Mirrors avo's own examples/dot structure.
package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

var unroll = 4

func main() {
	TEXT("dotInt8AVX2", NOSPLIT, "func(q []float32, code []int8) float32")
	q := Mem{Base: Load(Param("q").Base(), GP64())}
	c := Mem{Base: Load(Param("code").Base(), GP64())}
	n := Load(Param("q").Len(), GP64())

	acc := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		acc[i] = YMM()
		VXORPS(acc[i], acc[i], acc[i])
	}

	// Each unrolled lane consumes 8 int8 codes (1 byte each) and 8 float32
	// query values (4 bytes each) per iteration.
	blockitems := 8 * unroll
	codeBlockSize := blockitems
	queryBlockSize := 4 * blockitems

	Label("blockloop")
	CMPQ(n, U32(blockitems))
	JL(LabelRef("tail"))

	widened := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		widened[i] = YMM()
		VPMOVSXBD(c.Offset(8*i), widened[i]) // 8x int8 -> 8x int32
		VCVTDQ2PS(widened[i], widened[i])    // 8x int32 -> 8x float32
		VFMADD231PS(q.Offset(32*i), widened[i], acc[i])
	}

	ADDQ(U32(queryBlockSize), q.Base)
	ADDQ(U32(codeBlockSize), c.Base)
	SUBQ(U32(blockitems), n)
	JMP(LabelRef("blockloop"))

	// Scalar tail for the remaining < 8*unroll elements.
	Label("tail")
	tail := XMM()
	VXORPS(tail, tail, tail)

	Label("tailloop")
	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	codeInt := GP32()
	MOVBLSX(c, codeInt)

	codeFloat := XMM()
	VCVTSI2SSL(codeInt, codeFloat, codeFloat)

	qt := XMM()
	VMOVSS(q, qt)
	VFMADD231SS(qt, codeFloat, tail)

	ADDQ(U32(4), q.Base)
	ADDQ(U32(1), c.Base)
	DECQ(n)
	JMP(LabelRef("tailloop"))

	Label("reduce")
	for i := 1; i < unroll; i++ {
		VADDPS(acc[0], acc[i], acc[0])
	}

	result := acc[0].AsX()
	top := XMM()
	VEXTRACTF128(U8(1), acc[0], top)
	VADDPS(result, top, result)
	VADDPS(result, tail, result)
	VHADDPS(result, result, result)
	VHADDPS(result, result, result)
	Store(result, ReturnIndex(0))

	RET()

	Generate()
}
