Method

Each example is an annotated object from nuScenes-mini. The model uses saved
camera, radar, depth, and motion embeddings to predict whether the object is a
vehicle, pedestrian, or static obstacle.

The four training scenes are divided between four simulated clients. Each
client trains on its own scene, and the server combines the client models using
the number of objects on each client. FedProx adds a penalty that keeps a local
model from moving too far away from the global model.

Missing sensor tests are run after training. MCAR removes inputs randomly.
MNAR removes inputs more often from examples that are farther from their class
center in the training data.
