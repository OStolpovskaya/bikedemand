# My certification project

1. Run ModelGeneration. It will create model files under `models/RandomForest` directory
2. Run DemandPrediction. It will create output files under `result` directory
3. Run ModelStreamingText and then in terminal window use `nc -lk 9999` to send message:

```
2011-01-23 19:00:00,1,0,0,1,4.92,6.06,30,19.0012
2011-01-23 20:00:00,1,0,0,1,4.1,5.305,36,16.9979
2011-01-23 21:00:00,1,0,0,1,4.1,5.305,36,12.998
```

Output file is under `result` directory
Streaming screenshot is in `files` directory 