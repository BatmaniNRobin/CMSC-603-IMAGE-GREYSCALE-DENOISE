todo: main
main: main.cu
	nvcc -o main main.cu -lineinfo `pkg-config --cflags --libs opencv`
clean:
	rm main