import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """
        
        Initialize instance variables

        Argument(s)
        -----------
        
        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK


    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        # TODO: initialize skip_connect to be all zeros.

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        extended_symbols = [self.BLANK]
        skip_connect,prev_sym = [0],target[0]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)
            skip_connect.append(0) if prev_sym==symbol else skip_connect.append(1)
            skip_connect.append(0)
            prev_sym = symbol
        extended_symbols,skip_connect = np.array(extended_symbols).reshape((len(extended_symbols),)),np.array(skip_connect).reshape((len(extended_symbols),))
        return extended_symbols, skip_connect


    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))

        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # <---------------------------------------------
        alpha[0][0]=logits[0,extended_symbols[0]]
        alpha[0][1]=logits[0,extended_symbols[1]]
        for t in range(1,T):
            alpha[t][0] = alpha[t-1][0]*logits[t,extended_symbols[0]]
            for s in range(1,S):
                alpha[t,s] =alpha[t-1,s] + alpha[t-1,s-1]
                if s>1 and extended_symbols[s]!=extended_symbols[s-2]:
                    alpha[t,s]= alpha[t,s]+ alpha[t-1,s-2]
                alpha[t,s] *= logits[t,extended_symbols[s]]
  
        return alpha
        # raise NotImplementedError


    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities
        
        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))
        beta_t =  np.zeros((T,S))

        # -------------------------------------------->
        # TODO
        # <--------------------------------------------
        beta_t[T-1,S-1]= logits[T-1,extended_symbols[S-1]]
        beta_t[T-1,S-2]= logits[T-1,extended_symbols[S-2]]

        for t in reversed(range(T-1)):
            beta_t[t,S-1] =  beta_t[t+1,S-1]*logits[t,extended_symbols[S-1]]
            for s in reversed(range(S-1)):
                beta_t[t,s]=beta_t[t+1,s]+beta_t[t+1,s+1]
                if s<=S-3 and extended_symbols[s]!= extended_symbols[s+2]:
                    beta_t[t,s]+=beta_t[t+1,s+2]
                beta_t[t,s]*=logits[t,extended_symbols[s]]
        for s in reversed(range(S)):
              for t in reversed(range(T-1,-1,-1)):
                beta[t,s] = beta_t[t,s]/logits[t,extended_symbols[s]]

        return beta
        # raise NotImplementedError
        

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        [T, S] = alpha.shape
        gamma = np.zeros(shape=(T, S))
        sumgamma = np.zeros((T,))

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        for t in range(T):
            sumgamma[t] = 0
            for s in range(S):
                gamma[t,s] = alpha[t,s]*beta[t,s]
                sumgamma[t] += gamma[t,s]
            for s in range(S):
                gamma[t,s]= gamma[t,s] / sumgamma[t]
        return gamma
        # raise NotImplementedError


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------
        BLANK (int, optional): blank label index. Default 0.
        
        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()
        # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

        Computes the CTC Loss by calculating forward, backward, and
        posterior proabilites, and then calculating the avg. loss between
        targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        self.gamma=[]
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            
            target =self.target[batch_itr, :self.target_lengths[batch_itr]]
            logits= self.logits[:self.input_lengths[batch_itr], batch_itr, :]
            symbol,skip_connect= self.ctc.extend_target_with_blank( target)
   
            a= self.ctc.get_forward_probs(logits,symbol,skip_connect)
            b= self.ctc.get_backward_probs(logits,symbol,skip_connect)
   
            gamma=self.ctc.get_posterior_probs(a,b)

            self.gamma.append(gamma)
            self.extended_symbols.append(symbol)

            Time,Seq=len(logits),len(symbol)

            E_div=0
            for t in range(Time):
                for s in range(Seq):
                    k =self.gamma[batch_itr][t,s]*np.log(logits[t,symbol[s]])
                    E_div -= k 

            total_loss[batch_itr]=E_div

        total_loss = np.sum(total_loss) / B
        
        return total_loss
        # raise NotImplementedError
        

    def backward(self):
        """
        
        CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative 
        w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """
        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------

            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            target = self.target[batch_itr,:self.target_lengths[batch_itr]]
            logits = self.logits[:self.input_lengths[batch_itr],batch_itr,:]
            e_symbols,_ = self.ctc.extend_target_with_blank( target)
            Time= len(logits)
            Seq = len(e_symbols)
            for s in range(Seq):
                for t in range(Time):
                    k = self.gamma[batch_itr][t,s]/logits[t,e_symbols[s]]
                    dY[t,batch_itr,e_symbols[s]]-= k
        return dY
        # raise NotImplementedError
