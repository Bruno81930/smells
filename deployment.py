import libtmux as libtmux


class Approach:
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

        self.classifiers = ["rf", "svc", "nmp", "dt", "nb"]


class KNN(Approach):
    def __init__(self, dataset, ignore):
        super().__init__()
        self.dataset = dataset
        self.ignore = ignore
        self.projects = list(filter(lambda project: project not in self.ignore, self.projects))
        self.session_name = f"{dataset}_{self.__class__.__name__.lower()}"

    def __iter__(self):
        return self


smells_knn = KNN("smells", ['commons-csv', 'commons-cli', 'commons-email'])
server = libtmux.Server()
if server.has_session(smells_knn.session_name):
server.new_session(server)
