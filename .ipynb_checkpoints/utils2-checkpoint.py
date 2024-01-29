{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c1b6b6e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import stumpy\n",
    "import numpy as np\n",
    "import random\n",
    "import math\n",
    "import pickle\n",
    "import sys\n",
    "\n",
    "from statistics import mean\n",
    "from tqdm.auto import tqdm\n",
    "from multiprocessing import Pool\n",
    "\n",
    "import sklearn.metrics as metrics\n",
    "from sklearn.metrics import confusion_matrix, accuracy_score\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "198d8202",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Collects random samples from trace with id2 and computes the matrix profile of class1 compared with class 2\n",
    "\n",
    "Input: \n",
    "    trace1: packet traces from class 1\n",
    "    id2: id number for class 2 \n",
    "    num_traces: number of traces to select from class 2 (should be equal to class 1)\n",
    "    shapelet_size: length of shapelets\n",
    "    \n",
    "Output:\n",
    "    Matrix profile of trace1 compared with trace2\n",
    "'''\n",
    "def compare_profile(trace1, trace2, shapelet_size):\n",
    "    \n",
    "    length_diff = len(trace2) - len(trace1)\n",
    "    if(length_diff < 0):\n",
    "        trace2 = np.append(trace2, [np.nan] * abs(length_diff))\n",
    "        \n",
    "    #print(len(trace1))\n",
    "    #print(len(trace2))\n",
    "        \n",
    "    \n",
    "    c1_c2 = stumpy.stump(trace1, shapelet_size, trace2, ignore_trivial=False)[:, 0].astype(float)\n",
    "    c1_c2[c1_c2 == np.inf] = np.nan\n",
    "    \n",
    "    return c1_c2\n",
    "\n",
    "'''\n",
    "Compares a the matrix profile of a class trace with itself\n",
    "\n",
    "Input: \n",
    "    trace: packet traces from class 1\n",
    "    shapelet_size: length of shapelets\n",
    "    \n",
    "Output:\n",
    "    Matrix profile of trace compared with trace\n",
    "'''\n",
    "\n",
    "def same_profile(trace, shapelet_size):\n",
    "    \n",
    "    c1_c1 = stumpy.stump(trace, shapelet_size)[:, 0].astype(float)\n",
    "    c1_c1[c1_c1 == np.inf] = np.nan\n",
    "    \n",
    "    return c1_c1\n",
    "\n",
    "'''\n",
    "return indices of shapelet as one-hot encoded list\n",
    "'''\n",
    "def generate_shapelet(trace, diff, shapelet_size):\n",
    "    \n",
    "    idx = np.argmax(diff)\n",
    "    shapelet = np.asarray([1 if idx <= i < idx + shapelet_size else 0 for i in range(len(trace))])\n",
    "    \n",
    "    return shapelet\n",
    "\n",
    "'''\n",
    "Compute shapelet of greatest overlaps\n",
    "'''\n",
    "def find_overlap(trace_i, shapelets_i, shapelet_size):\n",
    "    #print(shapelets_i[0])\n",
    "    \n",
    "    merged_shapelets = np.sum(shapelets_i, axis=0)\n",
    "    \n",
    "    max_size = 0\n",
    "    start = 0\n",
    "    end = 0\n",
    "    \n",
    "    for i in range(0, len(merged_shapelets), shapelet_size):\n",
    "        current_size = np.sum(merged_shapelets[i:i+shapelet_size])\n",
    "        if current_size > max_size:\n",
    "            max_size = current_size\n",
    "            start = i\n",
    "            end = i + shapelet_size\n",
    "    \n",
    "    return trace_i[start:end]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "41239be1",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Generates a set of 100 shapelets for each class in samples\n",
    "\n",
    "Input:\n",
    "    num_traces = Number of traces per class\n",
    "    shapelet_size = Size of shapelets\n",
    "    save: save results to file?\n",
    "    filename: if save, name & location of output file\n",
    "\n",
    "Output:\n",
    "    list object containing shapelets for each class\n",
    "\n",
    "'''\n",
    "\n",
    "# !!! Choice of prototype - select min dist from each sample to all others\n",
    "# ! shapelet size\n",
    "\n",
    "# ! distace between comparing sample trace and shapelet (DTW? vs euclidean)\n",
    "\n",
    "# ! cross-validation on classifier (5 or 10 - fold) \n",
    "# ! classifier parameters\n",
    "\n",
    "# make results of changes at each stage for comparison (when writing paper)\n",
    "\n",
    "\n",
    "def generate_shapelets(shapelet_coeff):\n",
    "    shapelet_storage = []\n",
    "    \n",
    "    # loop over all classes (generate shapelet for each class)\n",
    "    for i in tqdm(range(100)):\n",
    "        \n",
    "        # get the chosen sample from trace i\n",
    "        trace_i = chosen_traces[i].astype('float64')\n",
    "        shapelet_size = math.floor(shapelet_coeff * len(trace_i))\n",
    "        \n",
    "        shapelets_i = np.zeros((100, len(trace_i)))\n",
    "        #print(shapelets_i.shape)\n",
    "        \n",
    "        # generate profile of i compared with itself\n",
    "        # length of sample is coeff* len*trace_i\n",
    "        ci_ci = same_profile(trace_i, shapelet_size)\n",
    "        \n",
    "        # loop over every other class and generate a profile for each one\n",
    "        for j in range(100):\n",
    "            # don't compare i with itself \n",
    "            if i == j:\n",
    "                continue\n",
    "            \n",
    "            trace_j = chosen_traces[j].astype('float64')\n",
    "            \n",
    "            # compute profile of i compared with j\n",
    "            ci_cj = compare_profile(trace_i, trace_j, shapelet_size)\n",
    "\n",
    "            # find largest value gap between other and i\n",
    "            diff_ci = ci_cj - ci_ci\n",
    "            \n",
    "            # generate best shapelet for i compared to j and store it in list\n",
    "            ci_shape = generate_shapelet(trace_i, diff_ci, shapelet_size)\n",
    "            shapelets_i[j] = ci_shape\n",
    "        \n",
    "        # compare shapelets between all classes and return the one which has the most overlap\n",
    "        # (i.e.) the shapelet that was chosen most between the 99 other classes\n",
    "        best_shapelet = find_overlap(trace_i, shapelets_i, shapelet_size)\n",
    "        # save to list\n",
    "        shapelet_storage.append(best_shapelet)\n",
    "    \n",
    "    return shapelet_storage   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "651bd8de",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Compute the minimum distance beteen data samples and shapelets\n",
    "Input:\n",
    "    data = list of individual packet traces\n",
    "    shapelets = list of shapelets\n",
    "Output:\n",
    "    minimum distance between each sample in data compared with each sample in shapelet\n",
    "    shape = (len(data),len(shapelets))\n",
    "'''\n",
    "def distance_to_shapelet(data, shapelets):\n",
    "    #data = np.asarray(data)\n",
    "    #print(len(data))\n",
    "    \n",
    "    # processed output data\n",
    "    data_out = np.zeros((len(data),len(shapelets)))\n",
    "    \n",
    "    # loop over each sample in the dataset\n",
    "    for i,sample in enumerate(tqdm(data)):\n",
    "        shapelet_score = np.empty(len(shapelets))\n",
    "        # for each shapelet, calculate distance and assign a score\n",
    "        for j,shapelet in enumerate(shapelets):\n",
    "            try:\n",
    "                dist = stumpy.mass(shapelet, sample)\n",
    "            except ValueError:\n",
    "                dist = stumpy.mass(sample, shapelet)\n",
    "            shapelet_score[j] = dist.min()\n",
    "        data_out[i] = shapelet_score\n",
    "    \n",
    "    return data_out\n",
    "\n",
    "'''\n",
    "Computes distances between input samples and shapelets, returns X for classifier\n",
    "Also cleans data and ensures no random errors due to length, NaN, etc...\n",
    "Underlying function that performs comparison is distance_to_shapelet\n",
    "Selects data samples (with replacement)\n",
    "note: some samples will always be bad so actual length of X is less\n",
    "\n",
    "Input:\n",
    "    num_traces = numner of traces to process\n",
    "    save = save output to file\n",
    "    filenames = tuple that represents (name of X file, name of y file)\n",
    "\n",
    "Output:\n",
    "    X values for classifier of shape (None, 100)\n",
    "    y values for classifier of shape (None, )\n",
    "'''\n",
    "\n",
    "def process_traces(num_traces, shapelets):\n",
    "    X, y = [], []\n",
    "\n",
    "    \n",
    "#     for i in range(num_traces):\n",
    "#         combo_trace = []\n",
    "#         combo_trace.append(random.choice(traces[random.randint(50,99)]))\n",
    "#         y_id = random.randint(0,49)\n",
    "#         combo_trace.append(random.choice(traces[y_id]))\n",
    "#         combo_trace.append(random.choice(traces[random.randint(50,99)]))\n",
    "#         out = np.concatenate((combo_trace[0],combo_trace[1],combo_trace[2]))\n",
    "        \n",
    "#         X.append(out)\n",
    "#         y.append(y_id)\n",
    "\n",
    "    # iterate over dictionary and re-format into X and y\n",
    "    for trace_id, trace_vals in traces.items():\n",
    "        for trace in trace_vals:\n",
    "            X.append(trace)\n",
    "            y.append(trace_id)\n",
    "    \n",
    "#     for i in range(num_traces):\n",
    "#         random_id = random.randrange(100)\n",
    "#         random_trace = random.choice(traces[random_id])\n",
    "#         X.append([random_trace])\n",
    "#         y.append(random_id)\n",
    "    \n",
    "    print(\"Size of X: \" + str(len(X)))\n",
    "    \n",
    "    \n",
    "    # process and remove useless entries (too short)\n",
    "    X = [np.asarray(trace).astype('float64') for trace in X]\n",
    "    X = [trace[~np.isnan(trace)] for trace in X]    \n",
    "\n",
    "    # compute distance between input trace and shapelet arrays\n",
    "    # return as new X\n",
    "\n",
    "    X = distance_to_shapelet(X, shapelets)\n",
    "    \n",
    "    return X, y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "485914fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Evaluate performance of sklearn classifier on data samples - 90/10 training testing split\n",
    "\n",
    "Input:\n",
    "    clf: sklearn classifier object\n",
    "    X: x values\n",
    "    y: y values\n",
    "    topk: k values for evaluation metrics\n",
    "Output:\n",
    "    list of length topk with accuracy for testing data\n",
    "'''\n",
    "\n",
    "def classifier_performance(clf, X, y, topk=[1,3,5]):\n",
    "    \n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)\n",
    "    \n",
    "    clf.fit(X_train, y_train)\n",
    "    y_prob = clf.predict_proba(X_test)\n",
    "    \n",
    "    scores = []\n",
    "    for k in topk:\n",
    "        correct = 0\n",
    "        for i in range(len(y_prob)):\n",
    "            ind = np.argpartition(y_prob[i], -k)[-k:]\n",
    "            if y_test[i] in ind:\n",
    "                correct += 1\n",
    "        scores.append(correct/len(y_prob))\n",
    "    \n",
    "    return scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "146c66f6",
   "metadata": {},
   "outputs": [],
   "source": [
    "'''\n",
    "Utility function for pipeline of evaluating different grid search parameters\n",
    "Output: a new file located in ../results/param1-val1_param2-val2_param3-val3\n",
    "        the file contains a pickled python object\n",
    "        with the scores for top-1, top-3, and top-5 classifier accuracy\n",
    "'''\n",
    "# note: python multiprocessing is really annoying to work with\n",
    "# function needs to be in a separate .py file which is imported\n",
    "# and function can only have 1 argument\n",
    "# list input which is immediately used for what would be the arguments\n",
    "def evaluate_parameters(arr):\n",
    "    \n",
    "    num_experiment = arr[0]\n",
    "    shapelet_coeff = arr[1]\n",
    "    \n",
    "    filename = '../results/shapelets/' + 'num=' + str(num_experiment) + 'size=' + str(shapelet_coeff)\n",
    "    #filename = '../results/data/trace_choice'\n",
    "    with open(filename, 'rb') as f:\n",
    "        shapelets = pickle.load(f)\n",
    "    \n",
    "    shapelets = [shapelet.astype('float64') for shapelet in shapelets]\n",
    "    \n",
    "    X, y = process_traces(num_samples, shapelets)\n",
    "    \n",
    "    filename = '../results/data/X/' + 'num=' + str(num_experiment) + 'size=' + str(shapelet_coeff)\n",
    "    \n",
    "    with open(filename, 'wb') as f:\n",
    "        pickle.dump(X, f)\n",
    "        \n",
    "    filename = '../results/data/y/' + 'num=' + str(num_experiment) + 'size=' + str(shapelet_coeff)\n",
    "    \n",
    "    with open(filename, 'wb') as f:\n",
    "        pickle.dump(y, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ae8b92b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "range(0, 12)\n"
     ]
    }
   ],
   "source": [
    "# SETUP\n",
    "\n",
    "global traces\n",
    "\n",
    "with open('../ipt_traces.npy', 'rb') as f:\n",
    "    traces = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2b06cd14",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]\n",
      "[[0, 1], [1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1], [7, 1], [8, 1], [9, 1], [10, 1], [11, 1]]\n"
     ]
    }
   ],
   "source": [
    "nums = list(range(12))\n",
    "size = [1]\n",
    "\n",
    "parameter_list = [[x,y] for x in nums for y in size]\n",
    "\n",
    "print(parameter_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "cac00d1f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.  0.5]\n",
      " [0.  0.6]\n",
      " [0.  0.7]\n",
      " [0.  0.8]]\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "3bbdb640ba2544139ec90e35aa4d6c7f",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "  0%|          | 0/100 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# PART 1\n",
    "\n",
    "print(parameter_list)\n",
    "\n",
    "for parameters in parameter_list:\n",
    "    coeff = parameters[1]\n",
    "    shapelets = generate_shapelets(coeff)\n",
    "    \n",
    "    filename = '../results/shapelets/' + 'num=' + str(parameters[0]) + 'size=' + str(parameters[1])\n",
    "    \n",
    "    with open(filename, 'wb') as f:\n",
    "        pickle.dump(shapelets, f)   "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0060f4ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "## PART 2\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    \n",
    "    global traces\n",
    "\n",
    "    with open('../nonzero_traces.npy', 'rb') as f:\n",
    "        traces = pickle.load(f)\n",
    "\n",
    "    from utils import evaluate_parameters\n",
    "    print(parameter_list)\n",
    "    \n",
    "    with Pool(6) as p:\n",
    "        p.map(evaluate_parameters, parameter_list)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "866a6d7c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[5, 1, 200000]]\n",
      "200000\n",
      "200000\n",
      "0.3907\n",
      "[0.0, 0.0, 0.0]\n"
     ]
    }
   ],
   "source": [
    "## PART 3\n",
    "\n",
    "print(parameter_list)\n",
    "\n",
    "for parameters in parameter_list:\n",
    "    \n",
    "    filename = '../results/data/X/' + 'num=' + str(parameters[0]) + 'size=' + str(parameters[1])\n",
    "    \n",
    "    with open(filename, 'rb') as f:\n",
    "        X = pickle.load(f)\n",
    "    \n",
    "    filename = '../results/data/y/' + 'num=' + str(parameters[0]) + 'size=' + str(parameters[1])\n",
    "    \n",
    "    with open(filename, 'rb') as f:\n",
    "        y = pickle.load(f)\n",
    "    \n",
    "    clf = RandomForestClassifier()\n",
    "    scores = classifier_performance(clf, X, y)\n",
    "    \n",
    "    print(scores)\n",
    "    \n",
    "    outfile_name = \"../results/scores/\" + 'num=' + str(parameters[0]) + 'size=' + str(parameters[1])\n",
    "    \n",
    "    with open(outfile_name, 'wb') as f:\n",
    "        pickle.dump(scores, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "id": "5437fecb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[3, 0.25, 400000]]\n",
      "[[3, 0.25, 400000]]\n"
     ]
    }
   ],
   "source": [
    "# PART 4\n",
    "\n",
    "print(parameter_list)\n",
    "\n",
    "for parameters in parameter_list:\n",
    "    \n",
    "    filename = '../results/data/X/' + 'num=' + str(parameters[0]) + 'size=' + str(parameters[1])\n",
    "    \n",
    "    with open(filename, 'rb') as f:\n",
    "        X = pickle.load(f)\n",
    "        \n",
    "    filename = '../results/data/y/' + 'num=' + str(parameters[0]) + 'size=' + str(parameters[1])\n",
    "    \n",
    "    with open(filename, 'rb') as f:\n",
    "        y = pickle.load(f)\n",
    "    \n",
    "    clf = RandomForestClassifier()\n",
    "    \n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)\n",
    "    clf.fit(X_train, y_train)\n",
    "    y_pred = clf.predict(X_test)\n",
    "    \n",
    "    matrix = confusion_matrix(y_test, y_pred)\n",
    "    scores = matrix.diagonal()/matrix.sum(axis=1)\n",
    "    \n",
    "    outfile_name = \"../results/scores_perclass/\" + 'num=' + str(parameters[0]) + 'size=' + str(parameters[1])\n",
    "    \n",
    "    with open(outfile_name, 'wb') as f:\n",
    "        pickle.dump(scores, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "id": "d49a7fe5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.83333333 0.91079812 0.97277228 0.93963255 0.93430657 0.93665158\n",
      " 0.89646465 0.77023499 0.85060241 0.92572944 0.92875318 0.89260143\n",
      " 0.90078329 0.79586563 0.91863517 0.89839572 0.92105263 0.92929293\n",
      " 0.87830688 0.91842105 0.95844156 0.91127098 0.95512821 0.80407125\n",
      " 0.83333333 0.95685279 0.8        0.97721519 0.774942   0.93253012\n",
      " 0.86956522 0.9079602  0.93333333 0.88664987 0.85532995 0.90074442\n",
      " 0.94444444 0.79600887 0.94736842 0.93573265 0.776      0.88308458\n",
      " 0.88516746 0.87823834 0.96524064 0.81266491 0.8989899  0.82442748\n",
      " 0.9119171  0.85714286 0.7893401  0.76570048 0.97435897 0.90675991\n",
      " 0.94827586 0.94482759 0.96244131 0.80569948 0.90339426 0.96933962\n",
      " 0.93931398 0.9382716  0.88148148 0.97375328 0.96009975 0.96287129\n",
      " 0.95939086 0.89473684 0.85750636 0.88967136 0.93857494 0.97368421\n",
      " 0.95640327 0.88235294 0.96049383 0.94987469 0.84486874 0.8973607\n",
      " 0.85900783 0.80154639 0.85023041 0.89027431 0.96938776 0.8778626\n",
      " 0.92443325 0.93633952 0.84556962 0.95823096 0.96560197 0.93924051\n",
      " 0.8880597  0.79466667 0.89367089 0.97342995 0.87135922 0.9041769\n",
      " 0.98271605 0.92271663 0.94117647 0.75970874]\n"
     ]
    }
   ],
   "source": [
    "print(scores)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d509ac65",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}