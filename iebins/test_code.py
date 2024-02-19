import torch


batch = 2
instances = 4 
labels = 5

# Generate random integers from 0 to 3

random_integers = torch.randint(-1, labels, (batch, instances))
#print(random_integers)
random_integers = random_integers.view(batch * instances,1)
valid_instances = torch.nonzero(random_integers != -1)

feature_map = (valid_instances.shape[0], 2)
random = torch.randn(feature_map)

feature_map_final = (batch, instances, 2)
feature_map_final = torch.zeros(feature_map_final)

feature_map_final = feature_map_final.view(batch * instances, 2)
print(feature_map_final)
feature_map_final[valid_instances[:,0]] = random
print(feature_map_final)
print(random)
print(valid_instances)
feature_map_final = feature_map_final.view(batch, instances, 2)
#print(valid_instances)
