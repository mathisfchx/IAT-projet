import subprocess

if __name__ == '__main__':
    subnet =  "112"
    #machines = ["01", "02", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    machines = ["14", "15", "16"]
    for i in range(len(machines)):
        train_id = subnet+machines[i]
        host = f"ldufour2@tc405-112-{machines[i]}.insa-lyon.fr"
        #get the 
        print("Getting npy for ", train_id)
        command = f"scp {host}:/home/ldufour2/Documents/4TC/IAT-projet/qweight/qagent_{train_id}*.npy ./qweight/"
        #execute the command
        subprocess.call(command, shell=True)
        #get all logs
        command = f"scp {host}:/home/ldufour2/Documents/4TC/IAT-projet/logs/*{train_id}*.* ./logs/{train_id}/"
        #execute the command
        print("Getting logs for ", train_id)
        subprocess.call(command, shell=True)
        