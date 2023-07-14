# Crossformer for MTS on Financial Data Series Mark I

> The Crossformer is split into the parts
>> 1. DSW. This is where we segment the timesteps, to encode timestep information, and then rearrange the matrix to encode spatial information
>> 2. TSA. This is where we use transformers to both capture temporal and spatial information
>> 3. HED. This is where we slowly get courser and courser information, and run the TSA over each granularity to pull out information
