import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        sym_set_dash = ['-'] +  self.symbol_set  
        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        for s in range(len(y_probs[0])):

            k = y_probs[:,s]
            path_prob=  path_prob * np.max(k)
            idx = np.argmax(k)
            decoded_path.append(sym_set_dash[idx])
            
        best_decoded_path = ""
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        # 5. Compress sequence (Inside or outside the loop)
        for s in range(len(decoded_path)):
            if s == 0 and decoded_path[s] != '-':
                best_decoded_path += decoded_path[s]
            elif decoded_path[s] != '-' and (decoded_path[s] != decoded_path[s-1] or decoded_path[s] != best_decoded_path[-1]):
                best_decoded_path += decoded_path[s]
        return best_decoded_path, path_prob



class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
        
    def initp_fsymbol(self,SymbolSet, y):            
        initb_pscore = {"": y[0]}
        initp_fblank = {""}
        initp_fsymbol = set(SymbolSet)
        init_pscore = {k: y[i+1] for i, k in enumerate(SymbolSet)}
        return initb_pscore, init_pscore,initp_fblank, initp_fsymbol
    
    def extend_blank(self,p_tblank,p_terminalsymbol, y,BlankPathScore,PathScore):
        upath_tblank,ublank_pscore = set(),{}
        for path in p_tblank:
            if path in upath_tblank:
                continue
            upath_tblank.add(path)
            ublank_pscore[path] = y[0]*BlankPathScore[path]
        for path in p_terminalsymbol:
            k = y[0]*PathScore[path]
            if path not in upath_tblank:
                ublank_pscore[path] = k
                upath_tblank.add(path) 
            else:
                ublank_pscore[path] = ublank_pscore[path] + k
        return ublank_pscore, upath_tblank
    
    def identicalmerging(self,p_tblank, BlankPathScore, p_terminalsymbol, PathScore):
        for p in p_tblank:
            score = BlankPathScore[p]
            MergedPaths = p_terminalsymbol.copy()
            FinalPathScore = PathScore.copy()
            if p not in MergedPaths:
                FinalPathScore[p] = score
                MergedPaths.add(p)
            else:
                FinalPathScore[p] += score
        return MergedPaths, FinalPathScore
    
    def symbolextends(self,p_tblank,p_terminalsymbol, SymbolSet, y,BlankPathScore,PathScore):
            UpdatedPathScore,UpdatedPathsWithTerminalSymbol ={},set()
            for j in SymbolSet:
                for path in p_tblank:
                    k = path + j
                    UpdatedPathScore[k] = y[SymbolSet.index(j)+1] * BlankPathScore[path]
                    UpdatedPathsWithTerminalSymbol.add(k)
                    
            for i, j in enumerate(SymbolSet): 
                for path in p_terminalsymbol:
                    z = path + j if (j != path[-1]) else path 
                    k=  PathScore[path] * y[i+1]
                    if z not in UpdatedPathsWithTerminalSymbol: 
                        UpdatedPathsWithTerminalSymbol.add(z) 
                        UpdatedPathScore[z] = k
                    else:
                        UpdatedPathScore[z] += k
            return UpdatedPathsWithTerminalSymbol, UpdatedPathScore
        
    def pruning(self,p_tblank,p_terminalsymbol, BlankPathScore, PathScore, BeamWidth):
            scorelist, PrunedPathScore, PrunedBlankPathScore,Prunedp_tblank,PrunedPathsWithTerminalSymbol = [], {},{},set(),set()
            scorelist.extend([BlankPathScore[p] for p in p_tblank])
            scorelist.extend([PathScore[p] for p in p_terminalsymbol])
            scorelist.sort(reverse=True)
            cutoff = scorelist[BeamWidth]  if BeamWidth < len(scorelist) else scorelist[-1]
            Prunedp_tblank = {p for p in p_tblank if BlankPathScore[p] > cutoff}
            PrunedBlankPathScore = {p: BlankPathScore[p] for p in Prunedp_tblank}
            PrunedPathsWithTerminalSymbol = {p for p in p_terminalsymbol if PathScore[p] > cutoff}
            PrunedPathScore = {p: PathScore[p] for p in PrunedPathsWithTerminalSymbol}            
            return  PrunedBlankPathScore, PrunedPathScore,Prunedp_tblank, PrunedPathsWithTerminalSymbol

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore, PathScore, BlankPathScore= None, None, {}, {}
        nblank_pscore, newpathscore,Newp_tblank, NewPathsWithTerminalSymbol = self.initp_fsymbol(self.symbol_set, y_probs[:,0])

        for i in  range(1,T):
            BlankPathScore, PathScore,p_tblank, p_terminalsymbol = self.pruning(Newp_tblank,NewPathsWithTerminalSymbol, 
                                                                               nblank_pscore,newpathscore, self.beam_width)
            
            nblank_pscore ,Newp_tblank= self.extend_blank(p_tblank, p_terminalsymbol, y_probs[:,i],BlankPathScore,PathScore)
            NewPathsWithTerminalSymbol, newpathscore = self.symbolextends(p_tblank, p_terminalsymbol, self.symbol_set, 
                                                                                    y_probs[:,i],BlankPathScore,PathScore)
       
        _, FinalPathScore = self.identicalmerging(Newp_tblank, nblank_pscore, NewPathsWithTerminalSymbol, newpathscore)  
        bestPath = max(FinalPathScore, key=FinalPathScore.get)
         
        
        return bestPath, FinalPathScore

    
    
