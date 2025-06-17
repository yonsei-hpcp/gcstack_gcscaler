TARGET := post-traces-processing

$(TARGET): post-traces-processing.cpp
	g++ -std=c++11 -o $@ $^ -lboost_iostreams

run: $(TARGET)
	./$(TARGET)

clean: 
	rm -f $(TARGET) *.o

