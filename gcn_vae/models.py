from gcn_vae.layers import *
from gcn_vae.metrics import *
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# --- AJOUT DU HYPERPARAMETRE BETA ---
# Si le flag n'existe pas encore (pour éviter les erreurs de redéfinition), on le définit
try:
    flags.DEFINE_float('beta', 0.2, r'Poids du graphe dynamique vs statique (0.0 = 100\% statique, 1.0 = 100\% dynamique)')
except tf.app.flags.FlagsError:
    pass # Déjà défini


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.auc = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() - DOIT RESTER GENERIQUE """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.auc = masked_auc(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
                
        
class Deep_GCN(GCN):
    def __init__(self, placeholders, input_dim, depth, **kwargs):
        self.depth = depth
        super(Deep_GCN, self).__init__(placeholders, input_dim, **kwargs)
        
    def _build(self):
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        for i in range(self.depth):
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden1,
                                               placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))


class DynamicGCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        # CORRECTION : On retire 'depth' des kwargs avant d'appeler super().__init__
        # car la classe parente 'Model' ne supporte pas cet argument.
        if 'depth' in kwargs:
            self.depth = kwargs.pop('depth')
        else:
            self.depth = 0
            
        super(DynamicGCN, self).__init__(**kwargs)
        
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        
        self.build()
    def _loss(self):
        # On applique le weight decay sur la 2ème couche (index 1) car c'est la conv de sortie
        if len(self.layers) > 1:
            for var in self.layers[1].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.auc = masked_auc(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        # 1. Graph Learner
        self.graph_learner = MetricGraphLearner(input_dim=self.input_dim,
                                                embedding_dim=128,
                                                act=tf.nn.tanh,
                                                logging=self.logging,
                                                sparse_inputs=True)
        
        # 2. Layer 1 (Hidden)s
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        # 3. Layer 2 (Output)
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def build(self):
        """ Surcharge Spécifique pour DynamicGCN """
        with tf.variable_scope(self.name):
            self._build()

        # 1. Apprendre le graphe dynamique
        self.adj_dynamic = self.graph_learner(self.inputs)
        
        # 2. Récupérer le graphe statique (passé dans support[0])
        self.adj_static = self.placeholders['support'][0]

        # --- COUCHE 1 ---
        x = self.inputs
        
        # Poids de la couche 1 (définis dans _build)
        w1 = self.layers[0].vars['weights_0']
        
        # Projection linéaire commune
        xw1 = dot(x, w1, sparse=True)
        
        # Branche Dynamique
        out_dyn_1 = tf.matmul(self.adj_dynamic, xw1)
        
        # Branche Statique
        out_stat_1 = dot(self.adj_static, xw1, sparse=True)
        
        # FUSION AVEC BETA
        # Si beta = 0.2, on prend 20% du dynamique et 80% du statique
        beta = FLAGS.beta
        layer1_pre_act = (beta * out_dyn_1) + ((1.0 - beta) * out_stat_1)
        
        # Activation + Dropout
        layer1_out = tf.nn.relu(layer1_pre_act)
        layer1_out = tf.nn.dropout(layer1_out, 1.0 - self.placeholders['dropout'])
        self.activations.append(layer1_out)

        # --- COUCHE 2 ---
        w2 = self.layers[1].vars['weights_0']
        xw2 = dot(layer1_out, w2, sparse=False)
        
        out_dyn_2 = tf.matmul(self.adj_dynamic, xw2)
        out_stat_2 = dot(self.adj_static, xw2, sparse=True)
        
        # Fusion couche de sortie
        layer2_out = (beta * out_dyn_2) + ((1.0 - beta) * out_stat_2)
        
        self.outputs = layer2_out
        
        # Fin standard
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        self._loss()
        self._accuracy()
        self.opt_op = self.optimizer.minimize(self.loss)
    def predict(self):
        return tf.nn.softmax(self.outputs)