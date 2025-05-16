GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM80 := -gencode=arch=compute_80,code=\"sm_80,compute_80\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"

CUOPTS =  $(GENCODE_SM80) $(GENCODE_SM70) $(GENCODE_SM89) $(GENCODE_SM90) 



CC := nvcc

INCLUDE := 
LIB := 

SRC = vimax3_s32_relu_global_bw.cu

EXE = vimax3_s32_relu_global_bw

NVCC_FLGAS = -Xptxas -dlcm=cg -Xptxas -dscm=wt

release:
	$(CC) $(NVCC_FLGAS) $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -lcudart

clean:
	rm -f *.o; rm -f $(EXE)

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)
