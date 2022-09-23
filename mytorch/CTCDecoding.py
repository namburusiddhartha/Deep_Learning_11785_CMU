import numpy as np


def clean_path(path):
	""" utility function that performs basic text cleaning on path """

	# No need to modify
	path = str(path).replace("'","")
	path = path.replace(",","")
	path = path.replace(" ","")
	path = path.replace("[","")
	path = path.replace("]","")

	return path


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

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        # 2. Iterate over symbol probabilities
        # 3. update path probability, by multiplying with the current max probability
        # 4. Select most probable symbol and append to decoded_path
        #print(y_probs[:, 0])
        length = len(y_probs[0])
        for l in range(length):
            max_prob = np.argmax(y_probs[:, l])
            if max_prob == 0:
                prob_cur = y_probs[max_prob, l]
            elif len(decoded_path) > 0 and self.symbol_set[max_prob-1] == decoded_path[-1]:
                prob_cur = y_probs[max_prob, l]
            else:
                prob_cur = y_probs[max_prob, l]
                decoded_path.append(self.symbol_set[max_prob-1])
            path_prob = path_prob * prob_cur                

        decoded_path = clean_path(decoded_path)

        return decoded_path, path_prob

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

        decoded_path = []
        sequences = [[list(), 1.0]]
        ordered = None

        best_path, merged_path_scores = None, None

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        #    - initialize a list to store all candidates
        # 2. Iterate over 'sequences'
        # 3. Iterate over symbol probabilities
        #    - Update all candidates by appropriately compressing sequences
        #    - Handle cases when current sequence is empty vs. when not empty
        # 4. Sort all candidates based on score (descending), and rewrite 'ordered'
        # 5. Update 'sequences' with first self.beam_width candidates from 'ordered'
        # 6. Merge paths in 'ordered', and get merged paths scores
        # 7. Select best path based on merged path scores, and return
        
        def Initialpaths(symbol_set, y):
            InitBlankPathScore = {} 
            InitPathScore = {}
            
            node = ""
            InitBlankPathScore[node] = y[0,0]
            InitPathWithFinalBlank = [node]
            
            InitalPathsWithFinalSymbol = []
            for s in range(len(symbol_set)):
                node = symbol_set[s]
                InitPathScore[node] = y[s + 1, 0]
                InitalPathsWithFinalSymbol.append(symbol_set[s])
            
            
            return InitPathWithFinalBlank, InitalPathsWithFinalSymbol, InitBlankPathScore, InitPathScore
        
        
        def ExtendWithBlank(pathsTB, pathsTS, y, PathScore, BlankPathScore):
            UpathsTB = []
            UBlankPathScore = {}
            
            for path in pathsTB:
                UpathsTB.append(path) 
                UBlankPathScore[path] = BlankPathScore[path]* y[0]
                
            for path in pathsTS:
                if path in UpathsTB:
                    UBlankPathScore[path] += PathScore[path]* y[0]
                else:
                    UpathsTB.append(path) 
                    UBlankPathScore[path] = PathScore[path] * y[0]
                    
            return UpathsTB, UBlankPathScore
        
        
        def ExtendWithSymbol(pathsTB, pathsTS, symbol_set, y, Pathscore, BlankPathScore):
            UpathsTS = []
            UBlankPathScore = {}
            
            for path in pathsTB:
                for s in range(len(symbol_set)):
                    sym = symbol_set[s]
                    newpath = path + sym
                    UpathsTS.append(newpath)
                    UBlankPathScore[newpath] = BlankPathScore[path] * y[s + 1]
            
            for path in pathsTS:
                for s in range(len(symbol_set)):
                    sym = symbol_set[s]
                    newpath = path if (sym == path[-1]) else path + sym
                    if newpath in UpathsTS:
                        UBlankPathScore[newpath] += Pathscore[path]* y[s + 1]
                    else:
                        UpathsTS.append(newpath)
                        UBlankPathScore[newpath] = Pathscore[path] * y[s + 1]
                        
                        
            return UpathsTS, UBlankPathScore
  
        
        def Prune(pathsTB, pathsTS, blankpathscore, pathscore, beam_width):
            pblankpathscore = {}
            ppathscore = {}
            
            scorelist = []
            for path in pathsTB:
                scorelist.append(blankpathscore[path])
                
            for path in pathsTS:
                scorelist.append(pathscore[path])
                
            scorelist.sort(reverse=True)
            
            cutoff = scorelist[beam_width - 1] if beam_width < len(scorelist) else scorelist[-1]
            
            PpathsTB = []
            for p in pathsTB:
                if blankpathscore[p] >= cutoff:
                    PpathsTB.append(p)
                    pblankpathscore[p] = blankpathscore[p]
                    
            PpathsTS = []
            for p in pathsTS:
                if pathscore[p] >= cutoff:
                    PpathsTS.append(p)
                    ppathscore[p] = pathscore[p]
                    
            
                    
            return PpathsTB, PpathsTS, pblankpathscore, ppathscore     
                 
        
        def MergeIdenticalPaths(pathsTB, pathsTS, blankpathscore, pathscore):
            MergedPaths = pathsTS
            FinalPathScore = pathscore
            
            for p in pathsTB:
                if p in MergedPaths:
                    FinalPathScore[p] += blankpathscore[p]
                else:
                    MergedPaths.append(p)
                    print(p)
                    FinalPathScore[p] += blankpathscore[p]
                    
                    
            return MergedPaths, FinalPathScore
            
            
        PathScore = {}
        BlankPathScore = {}

        
        length = len(y_probs[0])
        
        
        newpathsTB, newpathsTS, newblankpathscore, newpathscore = Initialpaths(self.symbol_set, y_probs)
        for t in range(1, length):
            pathsTB, pathsTS, BlankPathScore, PathScore = Prune(newpathsTB, newpathsTS, newblankpathscore, newpathscore, self.beam_width)
                        
            newpathsTB, newblankpathscore = ExtendWithBlank(pathsTB, pathsTS, y_probs[:, t], PathScore, BlankPathScore)
            
            newpathsTS, newpathscore = ExtendWithSymbol(pathsTB, pathsTS, self.symbol_set, y_probs[:,t], PathScore, BlankPathScore)
            
        merged_path_scores, FinalPathS = MergeIdenticalPaths(newpathsTB, newpathsTS, newblankpathscore, newpathscore)
                
        best_path = max(FinalPathS, key=FinalPathS.get)


        return best_path, FinalPathS
