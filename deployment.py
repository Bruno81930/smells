from time import sleep

import libtmux as libtmux
from alive_progress import alive_bar
from libtmux import Session
import numpy as np


class BaseApproach:
    def __init__(self):
        self.projects = ['accumulo', 'activemq', 'activemq-artemis', 'airavata', 'archiva', 'asterixdb', 'atlas',
                         'avro', 'beam', 'bookkeeper', 'calcite', 'camel', 'carbondata', 'cassandra', 'cayenne',
                         'clerezza', 'cocoon', 'commons-beanutils', 'commons-cli', 'commons-codec',
                         'commons-collections', 'commons-compress', 'commons-csv', 'commons-dbcp', 'commons-email',
                         'commons-io', 'commons-jexl', 'commons-lang', 'commons-math', 'commons-net',
                         'commons-validator', 'commons-vfs', 'continuum', 'crunch', 'curator', 'cxf', 'deltaspike',
                         'directory-kerby', 'directory-server', 'directory-studio', 'drill', 'flink', 'giraph',
                         'hadoop', 'hbase', 'helix', 'hive', 'isis', 'jackrabbit', 'jackrabbit-oak', 'jclouds', 'jena',
                         'johnzon', 'juneau', 'kafka', 'karaf', 'knox', 'kylin', 'lucene-solr', 'manifoldcf', 'maven',
                         'maven-surefire', 'metron', 'myfaces', 'myfaces-tobago', 'nifi', 'nutch', 'ofbiz',
                         'olingo-odata4', 'openjpa', 'openmeetings', 'opennlp', 'openwebbeans', 'parquet-mr', 'phoenix',
                         'plc4x', 'pulsar', 'qpid-jms', 'ranger', 'reef', 'roller', 'samza', 'santuario-java',
                         'servicecomb-java-chassis', 'shiro', 'storm', 'struts', 'syncope', 'systemml', 'tajo',
                         'tapestry-5', 'tez', 'tika', 'tinkerpop', 'tomcat', 'tomee', 'uima-ruta', 'wicket',
                         'xmlgraphics-fop', 'zeppelin']

        self.classifiers = ["rf", "svc", "mp", "dt", "nb"]
        self.approaches = {"std": False, "train": False, "test": False, "knn": True, "best": False, "elm": False}

        self.base_dir = "/home/machadob/PycharmProjects/smells"
        self.base_cmds = [
            "conda activate smells",
            "source env/bin/activate"
        ]

    @staticmethod
    def cmd(dataset, classifier, approach, project, nocache):
        project_cmd = ' '.join([f"-p {p}" for p in project])
        classifier_cmd = ' '.join([f"-c {c}" for c in classifier])
        nocache_cmd = '--nocache' if nocache else ''
        cmd = f"python crossproject.py -d {dataset} {classifier_cmd} -a {approach} {project_cmd} {nocache_cmd}"
        print(cmd)
        return cmd


class Approach(BaseApproach):
    def __init__(self, dataset, approach, ignore=None, ignore_clfs=None, enabled=True):
        super().__init__()
        self.dataset = dataset
        assert approach in list(self.approaches.keys()), "Wrong approach."
        self.approach = approach
        self.session_name = f"{dataset}_{approach}"
        self.nocache = self.approaches[approach]
        if ignore is not None:
            assert all(project in self.projects for project in ignore)
            self.projects = list(filter(lambda project: project not in ignore, self.projects))
        if ignore_clfs is not None:
            assert all(ignore_clf in self.classifiers for ignore_clf in ignore_clfs)
            self.classifiers = list(filter(lambda clf: clf not in ignore_clfs, self.classifiers))
        self.enabled = enabled

    def __call__(self, server):
        self.server = server
        session_name = f"{self.session_name}"
        print(session_name)
        if not self.server.has_session(session_name):
            self.session: Session = self.server.new_session(session_name)
        else:
            self.session = self.server.find_where({"session_name": session_name})

        self.window = self.session.select_window(0)
        if self.approach == "knn" or self.approach == "best":
            self.run(0, self.classifiers)
        else:
            [self.run(num, [classifier]) for num, classifier in enumerate(self.classifiers)]
        self.window.select_layout("tiled")

    def run(self, num, classifiers):
        vertical = num % 2 == 0
        pane = self.window.split_window(start_directory=self.base_dir, vertical=vertical)
        for cmd in self.base_cmds:
            pane.send_keys(cmd)
        cmd_args = {"dataset": self.dataset, "classifier": classifiers, "approach": self.approach,
                    "project": self.projects,
                    "nocache": self.nocache}
        keys = self.cmd(**cmd_args)
        if self.enabled:
            pane.send_keys(keys)


class KNN(Approach):
    def __init__(self, dataset, n_processes=10, ignore=None, enabled=True):
        super().__init__(dataset, "knn", ignore=ignore, enabled=enabled)
        self.dataset = dataset
        self.subprojects = np.array_split(self.projects, n_processes)
        self.session_names = [f"{dataset}_knn_{process}" for process in range(n_processes)]

    def __call__(self, server):
        self.server = server
        for name, projects in zip(self.session_names, self.subprojects):
            self.session_name = f"new_{name}"
            self.projects = projects
            super().__call__(server)


metrics_knn = KNN("metrics", ignore=['commons-cli', 'commons-codec', 'commons-collections', 'commons-compress', 'commons-csv', 'commons-dbcp', 'commons-email', 'commons-io', 'commons-jexl', 'commons-lang', 'commons-net', 'crunch', 'myfaces', 'myfaces-tobago', 'parquet-mr', 'samza', 'santuario-java', 'servicecomb-java-chassis', 'shiro'])
smells_knn = KNN("smells", n_processes=4, ignore=['accumulo', 'activemq', 'activemq-artemis', 'airavata', 'archiva', 'asterixdb', 'atlas', 'avro', 'beam', 'bookkeeper', 'calcite', 'camel', 'carbondata', 'cassandra', 'cayenne', 'clerezza', 'cocoon', 'commons-beanutils', 'commons-cli', 'commons-codec', 'commons-collections', 'commons-compress', 'commons-csv', 'commons-dbcp', 'commons-email', 'commons-io', 'commons-jexl', 'commons-lang', 'commons-math', 'commons-net', 'commons-validator', 'commons-vfs', 'continuum', 'crunch', 'curator', 'cxf', 'deltaspike', 'directory-kerby', 'directory-server', 'directory-studio', 'drill', 'flink', 'giraph', 'hadoop', 'hbase', 'helix', 'hive', 'isis', 'jackrabbit', 'jackrabbit-oak', 'jclouds', 'jena', 'johnzon', 'juneau', 'kafka', 'karaf', 'knox', 'kylin', 'lucene-solr', 'manifoldcf', 'maven', 'maven-surefire', 'metron', 'myfaces', 'myfaces-tobago', 'nifi', 'nutch', 'ofbiz', 'olingo-odata4', 'openjpa', 'openmeetings', 'opennlp', 'openwebbeans', 'parquet-mr', 'phoenix', 'plc4x', 'pulsar', 'qpid-jms', 'ranger', 'reef', 'roller', 'samza', 'santuario-java', 'servicecomb-java-chassis', 'shiro', 'storm', 'struts', 'syncope', 'systemml', 'tajo', 'tapestry-5', 'tez', 'tika', 'tinkerpop', 'tomcat', 'zeppelin'])
metrics_train = Approach("metrics",
                         "train",
                         ['accumulo', 'activemq', 'activemq-artemis', 'airavata', 'archiva', 'asterixdb', 'atlas',
                          'avro', 'beam', 'bookkeeper', 'calcite', 'camel', 'carbondata', 'cassandra', 'cayenne',
                          'clerezza',
                          'cocoon', 'commons-beanutils', 'commons-cli', 'commons-codec', 'commons-collections',
                          'commons-compress', 'commons-csv', 'commons-dbcp', 'commons-email', 'commons-io',
                          'commons-jexl', 'commons-lang', 'commons-math', 'commons-validator',
                          'commons-vfs', 'continuum', 'crunch', 'curator', 'cxf', 'deltaspike', 'directory-kerby',
                          'directory-server', 'directory-studio', 'drill', 'flink', 'giraph', 'hadoop', 'hbase',
                          'helix', 'hive', 'isis', 'jackrabbit', 'jackrabbit-oak', 'jclouds', 'jena', 'johnzon',
                          'juneau', 'kafka', 'karaf', 'knox', 'kylin', 'lucene-solr', 'manifoldcf', 'maven',
                          'maven-surefire', 'metron', 'myfaces', 'myfaces-tobago', 'nifi', 'nutch', 'ofbiz',
                          'olingo-odata4', 'openjpa', 'openmeetings', 'opennlp', 'openwebbeans', 'parquet-mr',
                          'phoenix', 'plc4x', 'pulsar', 'qpid-jms', 'ranger', 'reef', 'roller', 'samza',
                          'santuario-java', 'servicecomb-java-chassis', 'shiro', 'storm', 'struts', 'syncope',
                          'systemml', 'tajo', 'tapestry-5', 'tez', 'tika', 'tinkerpop', 'tomcat', 'tomee', 'uima-ruta',
                          'wicket', 'xmlgraphics-fop', 'zeppelin'],
                         ["rf", "svc", "dt", "nb"]
                         )
smells_metrics_std = Approach("smells_metrics", "std")
smells_metrics_train = Approach("smells_metrics", "train")
smells_metrics_test = Approach("smells_metrics", "test")
smells_metrics_knn = KNN("smells_metrics", ignore=['accumulo', 'activemq', 'activemq-artemis', 'airavata', 'archiva', 'asterixdb', 'atlas', 'avro', 'beam', 'bookkeeper', 'calcite', 'camel', 'carbondata', 'cassandra', 'cayenne', 'clerezza', 'cocoon', 'commons-beanutils', 'commons-cli', 'commons-codec', 'commons-collections', 'commons-compress', 'commons-csv', 'commons-dbcp', 'commons-email', 'commons-io', 'commons-jexl', 'commons-lang', 'commons-math', 'commons-net', 'commons-validator', 'commons-vfs', 'curator', 'cxf', 'deltaspike', 'directory-kerby', 'directory-server', 'johnzon', 'maven', 'maven-surefire', 'metron', 'myfaces', 'myfaces-tobago', 'olingo-odata4', 'shiro', 'storm', 'struts', 'syncope', 'tika'])
smells_metrics_best = Approach("smells_metrics", "best")
smells_elm = Approach("smells", "elm", ignore_clfs=["svc", "mp", "dt", "nb"])
metrics_elm = Approach("metrics", "elm", ignore=['accumulo', 'activemq', 'activemq-artemis', 'airavata', 'archiva', 'asterixdb', 'atlas', 'avro', 'beam', 'bookkeeper', 'calcite', 'camel', 'carbondata', 'cassandra', 'cayenne', 'clerezza', 'cocoon', 'commons-beanutils', 'commons-cli', 'commons-codec', 'commons-collections', 'commons-compress', 'commons-csv', 'commons-dbcp', 'commons-email', 'commons-io', 'commons-jexl', 'commons-lang', 'commons-math', 'commons-net', 'commons-validator', 'commons-vfs', 'continuum', 'crunch', 'curator', 'cxf', 'deltaspike', 'directory-kerby', 'directory-server', 'directory-studio', 'drill', 'flink', 'giraph', 'hadoop', 'hbase', 'helix', 'hive', 'isis', 'jackrabbit', 'jackrabbit-oak', 'jclouds', 'jena', 'johnzon', 'juneau', 'kafka', 'karaf', 'knox', 'kylin', 'lucene-solr', 'manifoldcf', 'maven', 'maven-surefire', 'metron', 'myfaces', 'myfaces-tobago', 'nifi', 'nutch', 'ofbiz', 'olingo-odata4', 'openjpa', 'openmeetings', 'opennlp', 'openwebbeans', 'parquet-mr', 'phoenix', 'plc4x', 'pulsar', 'qpid-jms', 'ranger', 'reef', 'roller', 'samza', 'santuario-java', 'servicecomb-java-chassis', 'storm', 'struts', 'syncope', 'systemml', 'tajo', 'tapestry-5', 'tez', 'tika', 'tomcat', 'tomee', 'uima-ruta', 'wicket', 'xmlgraphics-fop', 'zeppelin'], ignore_clfs=["svc", "mp", "dt", "nb"], enabled=True)
smells_metrics_elm = Approach("smells_metrics", "elm", ignore_clfs=["svc", "mp", "dt", "nb"])
special_knn = KNN("metrics", n_processes=20, ignore=['accumulo', 'bookkeeper', 'commons-beanutils', 'commons-cli', 'commons-codec', 'commons-collections', 'commons-compress', 'commons-csv', 'commons-dbcp', 'commons-email', 'commons-io', 'commons-jexl', 'commons-lang', 'commons-math', 'commons-net', 'commons-validator', 'commons-vfs', 'continuum', 'crunch', 'curator', 'directory-kerby', 'directory-server', 'helix', 'maven-surefire', 'metron', 'myfaces', 'myfaces-tobago', 'opennlp', 'openwebbeans', 'parquet-mr', 'phoenix', 'plc4x', 'roller', 'samza', 'santuario-java', 'servicecomb-java-chassis', 'shiro', 'storm', 'tika', 'tinkerpop', 'activemq', 'calcite', 'directory-studio', 'hive', 'kafka', 'nifi', 'pulsar', 'struts', 'tomcat'])
modules = [
    # metrics_knn,
    # smells_knn,
    # smells_metrics_std,
    # smells_metrics_train,
    # smells_metrics_test,
    # smells_metrics_knn,
    # smells_metrics_best,
    # smells_elm,
    # metrics_elm,
    # smells_metrics_elm
    # metrics_train
    special_knn
]
ISE2021 = libtmux.Server()
[module(ISE2021) for module in modules]
