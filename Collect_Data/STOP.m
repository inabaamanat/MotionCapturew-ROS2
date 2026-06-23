function [] = STOP()
pool = gcp;
cancelAll(pool.FevalQueue);
setTreadmill(0, 0,1*1000,1*1000)
end

