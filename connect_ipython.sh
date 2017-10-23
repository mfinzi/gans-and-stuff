machine=1
port_local=8889
port_remote=8888

usage()
{
  echo "usage: ./connect_ipython.sh [[[-n machine number] [-pl local port] [-pr remote port]] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -n | --nikola )         shift
                                machine=$1
                                ;;
        -pl | --port_local )    shift
                                port_local=$1
                                ;;
        -pr | --port_remote )   shift
                                port_remote=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
    esac
    shift
done

echo "Connecting to nikola0$machine at port $port_remote to port $port_local"

lsof -ti:$port_local | xargs kill -9
ssh -N -f -L localhost:$port_local:localhost:$port_remote pi49@nikola-compute0$machine.coecis.cornell.edu 
