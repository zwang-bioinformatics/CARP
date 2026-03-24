###################################

Please update the init.py with the correct paths

###################################

1. 
    IPKnot: https://github.com/satoken/ipknot
2. 
    NetSurfP (3.0): https://services.healthtech.dtu.dk/services/NetSurfP-3.0/
3. 
    AMIGOS: https://github.com/pylelab/AMIGOS
4.
    RNAView: https://github.com/rcsb/RNAView
5.
    LinearPartition: https://github.com/LinearFold/LinearPartition
6.
    MC-Annotate:  https://major.iric.ca/MajorLabEn/MC-Tools.html
    you can download and unzip the MC-Annotate.zip, and put the executable (unzip the .zip file) in {ROOT}/tools
    you must have MC-Annotate added to your path: 
        Put the MC-Annotate executable in {ROOT}/tools/
        Run `` export PATH="$PATH:{ROOT}/tools ``
        Run `` source .bashrc ``
        
###################################

Scoring Structures:

You should install [OpenStructure](https://openstructure.org/install), specifically version 2.11.1.
We used the OST provided [docker file](https://git.scicore.unibas.ch/schwede/openstructure/-/blob/master/docker/Dockerfile) and started the docker container using the following command:
```
docker run -d --name ost_worker --network host --entrypoint "" -v /home/asiciliano/CARP/tools/:/root/tools -v /home/asiciliano/CARP/data/targets/:/root/targets ost_2.11.1 tail -f /dev/null
```
You can verify the container is running by using:
```
docker ps
```
To terminate and remove the container:
```
docker stop ost_worker && docker rm ost_worker
```

###################################
