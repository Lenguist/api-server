input format:

{
  input_sentence: "[input string]"
}


docker build --tag basic-server-docker .
docker run basic-server-docker
docker run -d -p 5000:5000 lenguist/predict-from-allen:latest

docker build -t lenguist/predict-from-python:latest .

docker run -d -p 5000:5000 lenguist/ukramarly-app-v0.1:latest

docker push lenguist/predict-latest:latest

docker build -t lenguist/ukramarly-app-v0.1:latest .
