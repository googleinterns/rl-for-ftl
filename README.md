# rl-for-ftl

The environment defines a simple probabilistic environment which is meant to simulate an SSD with multiple write points and garbage collection.
The SSD is represented by an array in which each entry represents a write point and is capped by the amount of "space" available at that write point. 

User requests data to be written into SSD.

When writing data, an action corresponds to choosing a write-point to write the request. 

Each action is rewarded based on the instantaneous write-amplification obsereved. For example, if two blocks are written into a write point which 
is full, then the reward will be at most 1/2, since GC will need to remove two blocks in order to accomodate. Based on a probability "gc_probability"
the GC algorithm may actually clear up more space than just what is required, further lowering the instantaneous reward. 

Instantaneous write amplification is discounted if the user action corresponds to using the best write point (write point with most efficient GC),
or clustering data by putting it in the same write point as it was before. 

Tabular q-learning is applied to the system in order to learn how to optimally utilize both th GC discount and the clustering discount. 


