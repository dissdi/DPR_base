1. preprocessing 실행
2. main.py 실행
3. build faiss 실행

nohup sh -c "python main.py --config-name config && python build_faiss.py && python faiss_benchmark.py" > output.log &