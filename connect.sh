machine=1

usage()
{
  echo "usage: ./connect_ipython.sh [[-n machine number] | [-h]]"
}

while [ "$1" != "" ]; do
    case $1 in
        -n | --nikola )         shift
                                machine=$1
                                ;;
        -h | --help )           usage
                                exit
                                ;;
    esac
    shift
done

echo "Connecting to nikola0$machine"
ssh pi49@nikola-compute0$machine.coecis.cornell.edu
