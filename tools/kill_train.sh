ps axu | grep train | awk '{print $2}' | xargs kill -s 9
