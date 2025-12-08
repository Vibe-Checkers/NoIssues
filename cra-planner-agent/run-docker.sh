docker run -it --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e AZURE_OPENAI_API_KEY="CcEo1ZmEfZqREYvzL6NUhadCB5faLvVV9U62lwIPNnYHWQPeSFWoJQQJ99BJACfhMk5XJ3w3AAAAACOGNinG" \
  -e AZURE_OPENAI_ENDPOINT="https://no-issues-resource.openai.azure.com/" \
  -e AZURE_OPENAI_DEPLOYMENT="gpt-5-nano" \
  -v $(pwd)/library_links.txt:/app/library_links.txt \
  -v $(pwd)/parallel_empirical_results:/app/parallel_empirical_results \
  my-agent-image \
  library_links.txt --workers 8