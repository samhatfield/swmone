import numpy as np


class LocalNN:
    """
    Class for predicting the shallow water system using a neural net.
    The stencil size is passed as a constructor argument.
    """

    # Number of hidden layers and nodes per hidden layer
    n_hidden_layers = 2
    n_per_hidden_layer = 40

    # Fraction of raw training data to use for validation
    val_frac = 0.2

    def __init__(self, num_points, stencil):
        from util import build_model

        # Set stencil size
        self.stencil = stencil

        # Set output file (minus extension)
        self.out_file = f"{stencil}"

        # Build model for inference of interior of model domain
        self.model = build_model(
            stencil, 1,
            LocalNN.n_hidden_layers, LocalNN.n_per_hidden_layer
        )

        # Try loading weights file
        try:
            self.model.load_weights(f"models/localnn_{self.out_file}.hdf", by_name=False)
        except OSError as e:
            print(f"File models/{self.out_file}.hdf doesn't exist")
            print("Have you trained this model yet?")
            raise e

        # Store number of grid points
        self.N = num_points

        # Stores Adams-Bashforth steps
        self.η_tends = np.zeros((3,self.N))
        self.mode = 0

    """
    Advance variables by one time step.
    """
    def step(self, η):
        # self.η_tends = np.roll(self.η_tends, 1, axis=0)
        #
        # # Pad input for inferring grid points near boundaries
        # pad = int((self.stencil-1)/2)
        # η_pad = np.zeros((η.shape[0], η.shape[1]+2*pad, η.shape[2]))
        # η_pad[:,:pad,0]  = self.x_south[0]
        # η_pad[:,:pad,1]  = self.x_south[1]
        # η_pad[:,-pad:,0] = self.x_north[0]
        # η_pad[:,-pad:,1] = self.x_north[1]
        # η_pad[:,pad:-pad,:] = η
        #
        # # Prepare input array for neural net
        # infer_in = np.zeros((self.n_lon*self.n_lat,2*self.stencil**2))
        #
        # # Loop over all longitudes and latitudes
        # i = 0
        # for x in range(self.n_lon):
        #     for y in range(self.n_lat):
        #         infer_in[i,:] = InteriorNN.get_stencil(η_pad, x, y+pad, self.n_lon, self.stencil)
        #         i+=1
        #
        # # Normalize input
        # infer_in = InteriorNN.normalize_input(infer_in)
        #
        # # Predict new tendencies (tendencies include dt term)
        # tendencies = self.interior_model.predict(infer_in, batch_size=1)
        #
        # # Denormalize output
        # tendencies = InteriorNN.denormalize_output(tendencies)
        #
        # # Unpack tendencies
        # self.η_tends[0,:,:,0] = tendencies[:,0].reshape((self.n_lon,self.n_lat))
        # self.η_tends[0,:,:,1] = tendencies[:,1].reshape((self.n_lon,self.n_lat))
        #
        # # 3rd order Adams-Bashforth
        # if self.mode == 0:
        #     η_tend = self.η_tends[0,...]
        #     self.mode = 1
        # elif self.mode == 1:
        #     η_tend = 1.5*self.η_tends[0,...] - 0.5*self.η_tends[1,...]
        #     self.mode = 2
        # else:
        #     η_tend = (23.0/12.0)*self.η_tends[0,...] - (4.0/3.0)*self.η_tends[1,...] \
        #         + (5.0/12.0)*self.η_tends[2,...]
        #
        # # Step forward using forward Euler
        # return η + η_tend
        raise NotImplementedError

    """
    Train the neural net based on the input training data of η (streamfunction).
    """
    @staticmethod
    def train(stencil):
        from util import build_model, save_history
        from iris import load_cube

        # Attempt to load processed training data
        print("Attempting to load prepared training data")
        try:
            training_data   = np.load(f"training_data/localnn_{stencil}_training_data.npz")
            validation_data = np.load(f"training_data/localnn_{stencil}_validation_data.npz")

            # Split up training and validation data into input and output
            train_in, train_out  = training_data["train_in"], training_data["train_out"]
            val_in, val_out      = validation_data["val_in"], validation_data["val_out"]
        except FileNotFoundError:
            print("Prepared training data not found. Preparing now...")

            # Load training data
            η = load_cube("training_data/training_data.nc", ["eta"])

            # Transpose data so it's lon, lat, lev, time
            η.transpose()

            train_in, train_out, val_in, val_out = LocalNN.prepare_training_data(η.data, stencil)

            print("Training data prepared")

        print(f"Training with {train_in.shape[0]} training pairs,\
            dimensions: ({stencil}, 1)")

        # Build model for training
        model = build_model(
            stencil, 1,
            LocalNN.n_hidden_layers, LocalNN.n_per_hidden_layer
        )

        # Train!
        history = model.fit(train_in, train_out, epochs=20, batch_size=128,
                            validation_data=(val_in, val_out))

        # Output weights and diagnostics files
        save_history(f"models/localnn_{stencil}_history.txt", history)
        model.save_weights(f"models/localnn_{stencil}.hdf")

    """
    Prepare training data, including validation split.
    """
    @staticmethod
    def prepare_training_data(η, stencil):
        from numpy.random import shuffle

        # Get dimensions
        n_points, n_time = η.shape
        print(f"{n_points} points, {n_time} timesteps")

        # Compute number of training pairs
        # number of time steps (minus 1) * number of grid points
        n_train = (n_time - 1) * n_points

        # Define input and output arrays
        train_in_all  = np.zeros((n_train,stencil))
        train_out_all = np.zeros((n_train,1))

        # Prepare training data. Different grid points and time steps are considered as independent
        # training pairs.
        i = 0
        for t in range(n_time-1):
            for x in range(n_points):
                train_in_all[i,:]  = LocalNN.get_stencil(η[:,t], x, n_points, stencil)
                train_out_all[i,:] = η[x,t+1] - η[x,t]
                i += 1

        # Normalize training data
        train_in_all  = LocalNN.normalize_input(train_in_all)
        train_out_all = LocalNN.normalize_output(train_out_all)

        # Shuffle training data and extract validation set
        indices = np.arange(n_train, dtype=np.int32)
        shuffle(indices)
        train_indices = indices[:-int(LocalNN.val_frac*n_train)]
        val_indices   = indices[-int(LocalNN.val_frac*n_train):]
        train_in  = train_in_all[train_indices,:]
        train_out = train_out_all[train_indices,:]
        val_in    = train_in_all[val_indices,:]
        val_out   = train_out_all[val_indices,:]

        # Save training and validation data to file
        np.savez(f"training_data/localnn_{stencil}_training_data.npz",
                 train_in=train_in, train_out=train_out)
        np.savez(f"training_data/localnn_{stencil}_validation_data.npz",
                 val_in=val_in, val_out=val_out)

        return train_in, train_out, val_in, val_out

    """
    Extracts the nxn stencil corresponding to the requested longitude and latitude.
    e.g. if you request the 2nd longitude, 1st latitude (index starting from 0), and the stencil
    size is 3x3
    ---------------------------    -------
    |a|b|c|d|e|f|g|h|i|j|k|l|m|    |b|c|d|
    ---------------------------    -------
    |n|o|p|q|r|s|t|u|v|w|x|y|z| => |o|p|q|
    ---------------------------    -------
    |a|b|c|d|e|f|g|h|i|j|k|l|m|    |b|c|d|
    ---------------------------    -------
    """
    @staticmethod
    def get_stencil(full_array, x, n_points, stencil):
        include = int((stencil-1)/2)

        return full_array[np.array(range(x-include,x+include+1))%n_points]

    """
    Normalize the given input training data so values are between -1.0 and 1.0.
    """
    @staticmethod
    def normalize_input(training_data):
        # Maximum and minimum values of η based on a long run of the numerical model
        η_max, η_min = 1.213, -1.095

        # Normalize the training data
        return 2.0*(training_data - η_min)/(η_max - η_min) - 1.0

    """
    Normalize the given output training data so values are between -1.0 and 1.0.
    """
    @staticmethod
    def normalize_output(training_data):
        # Maximum and minimum values of tendencies of η based on a long run of the numerical model
        η_max, η_min = 0.422, -0.026

        # Normalize the training data
        return 2.0*(training_data - η_min)/(η_max - η_min) - 1.0

    """
    Denormalize the given output.
    """
    @staticmethod
    def denormalize_output(output):
        # Maximum and minimum values of tendencies of η based on a long run of the numerical model
        η_max, η_min = 0.422, -0.026

        # Denormalize the output
        return (η_max - η_min)*(1.0 + output)/2.0  + η_min
