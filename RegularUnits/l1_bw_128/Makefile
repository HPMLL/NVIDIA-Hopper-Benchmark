GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"

CUOPTS =  $(GENCODE_SM61) $(GENCODE_SM70) $(GENCODE_SM89) $(GENCODE_SM90) 


CC := nvcc

INCLUDE := 
LIB := 

SRC = l1_bw_128.cu

EXE = l1_bw_128

release:
	$(CC) $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -lcudart

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)
