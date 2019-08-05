import os
import hashlib
import json
import ast
from datetime import datetime
import shutil
import pandas as pd

from mead.downloader import DataDownloader
from baseline.utils import listify
from xpctl.utils import write_config_file
from xpctl.xpclient.models import Dataset, Datafile
from xpctl.xpclient.rest import ApiException

pd.set_option('display.expand_frame_repr', False)
pd.set_option("display.max_rows", None)


def pack_result(results):
    """ List of results to event data"""
    metrics = set([r.metric for r in results])
    d = {metric: [] for metric in metrics}
    for result in results:
        d[result.metric].append(result.value)
    return pd.DataFrame(d)


def pack_aggregate_result(results, aggregate_fns):
    metrics = [r.metric for r in results]
    metrics = set(metrics)
    d = {metric: {} for metric in metrics}
    for result in results:
        for value in result.values:
            aggregate_fn = value.aggregate_fn
            if aggregate_fn not in aggregate_fns:
                continue
            score = value.score
            if aggregate_fn in d[result.metric]:
                d[result.metric][aggregate_fn].append(score)
            else:
                d[result.metric][aggregate_fn] = [score]

    dfs = {metric: pd.DataFrame.from_dict(d[metric])[list(aggregate_fns)] for metric in metrics}
    return pd.concat(dfs.values(), axis=1, keys=dfs.keys())


def get_prop_value(exp, prop_name):
    return exp.__dict__.get('_'+prop_name)


def insert_in_df(prop_name_loc, df, exp):
    for prop_name, location in prop_name_loc.items():
        df.insert(location, column=prop_name, value=[get_prop_value(exp, prop_name)]*len(df))


def experiment_to_df(exp, prop_name_loc={}, event_type='test_events', sort=None):
    prop_name_loc = {'sha1': 0, 'id': 1, 'username':  2} if not prop_name_loc else prop_name_loc
    if event_type == 'train_events' and exp.train_events:
        result_df = pack_result(exp.train_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if event_type == 'valid_events' and exp.valid_events:
        result_df = pack_result(exp.valid_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if event_type == 'test_events' and exp.test_events:
        result_df = pack_result(exp.test_events)
        insert_in_df(prop_name_loc, result_df, exp)
    if sort is not None:
        result_df.sort_values(by=sort, inplace=True)
    return result_df


def experiment_aggregate_to_df(exp_agg, prop_name_loc, event_type='test_events',
                               aggregate_fns=['min', 'max', 'avg', 'std']):
    event_dfs = []
    if event_type == 'train_events':
        train_df = pack_aggregate_result(exp_agg.train_events, aggregate_fns)
        insert_in_df(prop_name_loc, train_df, exp_agg)
        event_dfs.append(train_df)
    if event_type == 'valid_events':
        valid_df = pack_aggregate_result(exp_agg.valid_events, aggregate_fns)
        insert_in_df(prop_name_loc, valid_df, exp_agg)
        event_dfs.append(valid_df)
    if event_type == 'test_events':
        test_df = pack_aggregate_result(exp_agg.test_events, aggregate_fns)
        insert_in_df(prop_name_loc, test_df, exp_agg)
        event_dfs.append(test_df)
    result_df = pd.DataFrame()
    for event_df in event_dfs:
        result_df = result_df.append(event_df)
    return result_df


def experiment_aggregate_list_to_df(exp_aggs, event_type='test_events',
                                    aggregate_fns=['min', 'max', 'avg', 'std']):
    result_df = pd.DataFrame()
    prop_name_loc = {'dataset': 0, 'sha1': 1, 'num_exps': 2}
    for exp_agg in exp_aggs:
        result_df = result_df.append(experiment_aggregate_to_df(exp_agg, prop_name_loc, event_type, aggregate_fns))
    return result_df


def experiment_list_to_df(exps, prop_name_loc={}, event_type='test_events'):
    result_df = pd.DataFrame()
    prop_name_loc = {'sha1': 0, 'id': 1, 'username':  2} if not prop_name_loc else prop_name_loc
    for exp in exps:
        result_df = result_df.append(experiment_to_df(exp, prop_name_loc, event_type, sort=None))
    return result_df


def write_to_config_file(config_obj, filename):
    write_config_file(config_obj, filename)


def task_summary_to_df(tasksummary):
    def identity(x): return x
    summary = tasksummary.summary
    all_results = []
    for dataset in summary:
        for user, num_exps in summary[dataset]:
            all_results.append([user, dataset, num_exps])
    return pd.DataFrame(all_results, columns=['user', 'dataset', 'num_exps']).groupby(['user', 'dataset'])\
        .agg([identity]).rename(columns={'identity': ''})


def task_summaries_to_df(tasksummaries):
    def identity(x): return x
    all_results = []
    for tasksummary in tasksummaries:
        task = tasksummary.task
        summary = tasksummary.summary
        for dataset in summary:
            for user, num_exps in summary[dataset]:
                all_results.append([task, user, dataset, num_exps])
    return pd.DataFrame(all_results, columns=['task', 'user', 'dataset', 'num_exps']).groupby(['task', 'user', 'dataset'])\
        .agg([identity]).rename(columns={'identity': ''})


def read_config_stream(config):
    try:
        return json.loads(config)
    except json.decoder.JSONDecodeError:
        try:
            return ast.literal_eval(config)
        except ValueError:
            return None


def get_sha1(location):
    buf_size = 65536  # 64kb
    sha1 = hashlib.sha1()
    with open(location, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def safe_get_list(l, idx):
    try:
        return l[idx]
    except IndexError:
        return None
    

def has_file_changed(existing_data_file_obj, new_file_location):
    """
    existing data file object has a sha1. Does the new file location sha1 match with that?
    :param existing_data_file_obj:
    :param new_file_location:
    :return:
    """
    if existing_data_file_obj is None:
        return True
    existing_sha1 = existing_data_file_obj.sha1
    new_sha1 = get_sha1(new_file_location)
    return existing_sha1 != new_sha1


def create_datafile_object(location, writer):
    """
    We have to create a new DataFile object. If the writer is None, we will not try to push the
    file to a centralized location. Else we will.
    :param location:
    :param writer:
    :return:
    """
    file_sha1 = get_sha1(location)
    if writer is None:
        return Datafile(location=location, sha1=file_sha1)
    else:
        return Datafile(location=writer.write(location), sha1=file_sha1)


def create_datafiles_if_changed(database_data_files, user_locations, writer):
    """
    In database we recorded some files for this dataset. The user has provided some new locations. If the files
    have changed, we will need to update the database. This method sends back the new records to
    update the database with.
    :param database_data_files:
    :param user_locations:
    :return:
    """
    file_changed = False
    mod_data_files = []
    for index in range(len(user_locations)):
        existing_data_file_obj = safe_get_list(database_data_files, index)
        new_file_location = user_locations[index]
        file_changed = has_file_changed(existing_data_file_obj, new_file_location)
        if file_changed:
            mod_data_files.append(create_datafile_object(new_file_location, writer))
        else:
            mod_data_files.append(existing_data_file_obj)
    return mod_data_files, file_changed


def parse_date_time(s):
    KNOWN_FMTS = ['%Y%m%d%H%M%S']
    for fmt in KNOWN_FMTS:
        try:
            dt = datetime.strptime(s, fmt)
            return dt
        except:
            continue
    raise Exception("Couldn't parse datestamp {}".format(s))


def get_last_dataset(dataset_names):
    last_date = parse_date_time('1900010100001')
    last_dataset = None
    for dataset_name in dataset_names:
        if ':' in dataset_name:
            dt = parse_date_time(dataset_name.split(':')[-1])
            if dt > last_date:
                last_date = dt
                last_dataset = dataset_name
    if last_dataset is None:
        raise Exception("No dataset could be found")
    return last_dataset
    
    
def create_xpctl_dataset(dataset_desc, data_cache, api, force=False, writer=None):
    """
    create a single Dataset object that will be inserted in xpctl server. dataset_desc is a dictionary
    `{'train_file', 'valid_file', 'test_file', 'label'}`. If the label (dataset name) does not exist in the database, it
     will push the files to a commonly accessible place through a writer object.
    If the label already exists in the database, we will see (if the force option is enabled) if the files have changed. If yes, we will insert a new record:
    if your previous dataset was sst2:2018:09:11, the timestamp will be replaced by today's timestamp.
    :param dataset_desc: a dataset description dictionary {'train_file', 'valid_file', 'test_file', 'label'}
    :param data_cache: data cache for downloading files. required for calculating sha1
    :param api: xpctl api
    :param force:
    :return:
    """
    # does this dataset already exist in the database?
    dataset_name = dataset_desc['label']
    results = []
    try:
        results = api.get_datasets(name=dataset_name)
        exists = True
    except ApiException:
        exists = False
    if exists and not force:  # nothing to do here
        return 'The dataset name already exists. use --force if the files have changed'
    
    # look at what the user has supplied
    user_dataset = DataDownloader(dataset_desc, data_cache).download()
    user_train_files = listify(user_dataset['train_file'])
    user_valid_files = listify(user_dataset['valid_file'])
    user_test_files = listify(user_dataset['test_file'])
    for user_file in user_train_files + user_valid_files + user_test_files:
        if not os.path.exists(user_file):
            return 'User provided file {} does not exist, can not change existing record'.format(user_file)
    if exists:  # though the dataset name exists, the user wants to change the record. check if anything has changed
        # first check if the provided dataset can be uniquely identified. i.e., dataset name is not x when x:y:z exists.
        # if yes, use the last one
        if len(results) != 1:
            last_dataset_name = get_last_dataset([result.name for result in results])
            result = [result for result in results if result.name == last_dataset_name][0]
        else:
            result = results[0]
        # create new DataFile objects if anything has changed.
        mod_train_files, train_changed = create_datafiles_if_changed(result.train_files, user_train_files, writer)
        mod_valid_files, valid_changed = create_datafiles_if_changed(result.valid_files, user_valid_files, writer)
        mod_test_files, test_changed = create_datafiles_if_changed(result.test_files, user_test_files, writer)
        if train_changed or valid_changed or test_changed:
            if ':' in dataset_desc['label']:
                dataset_name = ':'.join(dataset_desc['label'].split(':')[:-1])  # strip the previous timestamp
            else:
                dataset_name = dataset_desc['label']
            new_dataset_name = '{}:{}'.format(dataset_name, datetime.now().strftime("%Y%m%d%H%M%S"))
            return Dataset(
                name=new_dataset_name,
                train_files=mod_train_files,
                valid_files=mod_valid_files,
                test_files=mod_test_files
            )
        else:
            return 'The dataset {} has not changed. Not inserting user supplied dataset.'.format(dataset_desc['label'])
    else:  # the dataset name does not exist in the database. add a new record.
        mod_train_files, train_changed = create_datafiles_if_changed([None], user_train_files, writer)
        mod_valid_files, valid_changed = create_datafiles_if_changed([None], user_valid_files, writer)
        mod_test_files, test_changed = create_datafiles_if_changed([None], user_test_files, writer)
        # all files should be changed because we are inserting a new record
        assert train_changed
        assert valid_changed
        assert test_changed
        dataset_name = dataset_desc['label']
        return Dataset(
            name=dataset_name,
            train_files=mod_train_files,
            valid_files=mod_valid_files,
            test_files=mod_test_files
        )


class Writer(object):
    """
    writes a file at a location.
    """
    def __init__(self):
        super(Writer, self).__init__()
        
    def exists(self, filename):
        """
        does the file exist at this location
        :param filename:
        """
        pass
    
    def write(self, _input, **kwargs):
        """
        write the file at input location at a pre-specified space and return the new file location
        :param _input:
        """
        pass


class LocalWriter(object):
    """
    Writes a file to a local directory.
    """
    def __init__(self, data_cache='/data/xpctl-data-cache'):
        super(LocalWriter, self).__init__()
        if not os.path.exists(data_cache):
            os.mkdir(data_cache)
        self.data_cache = data_cache

    def exists(self, filename):
        return os.path.exists(filename)
    
    def write(self, _input, **kwargs):
        """
        check by sha1 if you have to write or not
        :param _input:
        :param kwargs:
        :return:
        """
        sha1 = kwargs.get('sha1', None)
        if sha1 is None:
            sha1 = get_sha1(_input)
        write_to = os.path.join(self.data_cache, sha1)
        if not self.exists(write_to):
            shutil.copyfile(_input, write_to)
        return write_to


class RemoteWriter(object):
    """
    Writes a file to a remote directory.
    """
    def __init__(self, hostname, port=22, username=None, password=None, data_cache='/data/xpctl-data-cache'):
        super(RemoteWriter, self).__init__()
        import paramiko
        from paramiko import SSHClient
        import socket
        from scp import SCPClient
        self.scp = None
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.host = hostname
        self.data_cache = data_cache
        try:
            self.ssh.connect(hostname=hostname, port=port, username=username, password=password)
        except paramiko.ssh_exception.BadHostKeyException:
            self.status = "server host key can not be verified"
            return
        except paramiko.ssh_exception.AuthenticationException:
            self.status = "authentication failed"
            return
        except paramiko.ssh_exception.SSHException:
            self.status = "error establishing an SSH session"
            return
        except socket.error:
            self.status = "connection failed"
            return
        self.scp = SCPClient(self.ssh.get_transport())

    def exists(self, filename):
        import errno
        sftp = self.ssh.open_sftp()
        try:
            sftp.stat(filename)
            return True
        except IOError as e:
            if e.errno == errno.ENOENT:
                return False
            raise RuntimeError('unknown error checking for file at host {}'.format(self.host))
            
    def write(self, _input, **kwargs):
        """
        check by sha1 if you have to write or not
        :param _input:
        :param kwargs:
        :return:
        """
        sha1 = kwargs.get('sha1', None)
        if sha1 is None:
            sha1 = get_sha1(_input)
        write_to = os.path.join(self.data_cache, sha1)
        if not self.exists(write_to):
            self.scp.put(_input, write_to)
        return write_to
