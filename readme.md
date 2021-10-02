input format:

{
  input_sentence: "[input string]"
}



docker build -t lenguist/server-1.0:latest .

docker run -d -p 5000:5000 lenguist/server-1.0:latest

docker push lenguist/server-1.0:latest
