```
ACCOUNT=875518171890
REGION=eu-central-1
REGISTRY=bees-wasps-images
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}

TAG=bees-wasps-model-v2-001
REMOTE_URI=${PREFIX}:${TAG}
```
