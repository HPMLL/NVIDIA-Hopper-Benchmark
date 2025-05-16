BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin

all: 	
	mkdir -p $(BIN_DIR)
	cd alu_lat_double;		make;	mv alu_lat_double $(BIN_DIR)
	cd alu_lat_float;		make;	mv alu_lat_float $(BIN_DIR)
	cd alu_lat_half;		make;	mv alu_lat_half $(BIN_DIR)
	cd alu_lat_int32;		make;	mv alu_lat_int32 $(BIN_DIR)
	cd l1_bw_32f;			make;	mv l1_bw_32f $(BIN_DIR)
	cd l1_bw_64f; 			make;	mv l1_bw_64f $(BIN_DIR)
	cd l1_bw_128;			make;	mv l1_bw_128 $(BIN_DIR)
	cd l1_lat; 				make;	mv l1_lat $(BIN_DIR)
	cd l2_bw_32f;  			make;	mv l2_bw_32f $(BIN_DIR)
	cd l2_bw_64f; 			make;	mv l2_bw_64f $(BIN_DIR)
	cd l2_bw_128; 			make;	mv l2_bw_128 $(BIN_DIR)
	cd l2_lat; 				make;	mv l2_lat $(BIN_DIR)
	cd MaxFlops; 			make;	mv MaxFlops $(BIN_DIR)
	cd MaxFlops_16; 		make;	mv MaxFlops_16 $(BIN_DIR)
	cd MaxFlops_64; 		make;	mv MaxFlops_64 $(BIN_DIR)
	cd MaxFlops_int32; 		make;	mv MaxFlops_int32 $(BIN_DIR)
	cd MaxFlops_int64; 		make;	mv MaxFlops_int64 $(BIN_DIR)
	cd mem_bw; 				make;	mv mem_bw $(BIN_DIR)
	cd mem_lat; 			make;	mv mem_lat $(BIN_DIR)
	cd shared_lat; 			make;	mv shared_lat $(BIN_DIR)
	cd shared_bw; 			make;	mv shared_bw $(BIN_DIR)
	cd shared_bw_64; 		make;	mv shared_bw_64 $(BIN_DIR)
	

clean:
	cd $(BIN_DIR); rm -f *
