#this code is used to merge AI from npy files
import numpy as np

def mean_merging(Q):
    print("mean merging")
    AI = np.zeros((64,2,20,4))
    for x in range(64):
        for direction in range(2):
            for y in range(20):
                for action in range(4):
                    for i in range(len(Q)):
                        try :
                            AI[x][direction][y][action] += Q[i][x][direction][y][action]
                        except:
                            print("x:", x, "direction:", direction, "y:", y, "action:", action, "i:", i)
                            exit()
    return AI/len(Q)

def max_maring(Q):
    print("max merging")
    AI = np.zeros((64,2,20,4))
    for x in range(64):
        for direction in range(2):
            for y in range(20):
                max_val = -1
                for action in range(4):
                    for i in range(len(Q)):
                        try :
                            if Q[i][x][direction][y][action] > max_val:
                                max_val = Q[i][x][direction][y][action]
                                AI[x][direction][y][action] = Q[i][x][direction][y][action]
                        except:
                            print("x:", x, "direction:", direction, "y:", y, "action:", action, "i:", i)
                            exit()
    return AI

def sum_merging(Q):
    print("sum merging")
    AI = np.zeros((64,2,20,4))
    for x in range(64):
        for direction in range(2):
            for y in range(20):
                for action in range(4):
                    for i in range(len(Q)):
                        try :
                            AI[x][direction][y][action] += Q[i][x][direction][y][action]
                        except:
                            print("x:", x, "direction:", direction, "y:", y, "action:", action, "i:", i)
                            exit()
    return AI

def merge_npy(method):
    train_ids = ["11206", "11208", "11216"]
    Q = [np.zeros((64,2,20,4)) for _ in range(len(train_ids))]
    for train_id in train_ids:
        #load the npy file
        try:
            Q.append(np.load("qweight/qagent_"+train_id+".npy"))
        except:
            print("File not found")
            return
    #merge the npy files based on the method
    if method == "mean":
        AI = mean_merging(Q)
    elif method == "max":
        AI = max_maring(Q)
    elif method == "sum":
        AI = sum_merging(Q)
    #if AI is defined
    if AI is not None:
        np.save(f"qweight/qagent_{method}_merged.npy", AI)



def main():
    merge_npy("mean")
    merge_npy("max")
    merge_npy("sum")
    return 0

if __name__ == "__main__":
    main()