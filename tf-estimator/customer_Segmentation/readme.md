## Customer Segmentation ##

This Repo contains Customer Segmentation code implemented using Tensorflow Estimator API.

To create a dummy dataset, run the following command

` python data_generator.py`  which creates a dummy data enough to work on KMeans Clustering Algorithm.

For prediction from the CloudML 

`gcloud ml-engine predict --model <model_name> --version <model_version> --json-instances test.json project=<project_name>`

