image='tensorflow/tensorflow:1.15.5-gpu'
container_name=$1
cmd='bash'
nvidia-docker run  -itd --name $container_name  \
                   --restart=always \
                   --network host \
                   --shm-size 32g \
                   -v $PWD:/workspace \
                   $image  \
                   $cmd
