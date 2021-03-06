import sys
import logging
import io


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def create_file_name(args, q):
    file_name = "MLP-L{}-Q{}-K{}".format(args.layers, q, args.k)
    return file_name


def create_file_name_no_q(args):
    file_name = "MLP-L{}-K{}".format(args.layers, args.k)
    return file_name


def init_log(args):
    file_name = create_file_name_no_q(args)
    log_file = args.data_dir + "/models/" + file_name + ".log"
    print("Writing log to file", log_file, flush=True)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S', filename=log_file)


def my_print(*objects, sep=' ', end='\n'):
    print(*objects, sep=sep, end=end, file=sys.stdout, flush=True)
    logging.info(print_to_string(*objects, sep, end=''))
