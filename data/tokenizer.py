from typing import List
from collections import defaultdict

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        results=[]
        tokens=list(corpus)
        for _ in range(num_merges):
            i=0
            count=defaultdict(int)
            n=len(tokens)
            while i<n-1:
                count[(tokens[i],tokens[i+1])]+=1
                i+=1
            candidate=min(count,key=lambda x: (-count[x],x))
            j=0
            new_tokens=[]
            while j<n-1:
                if (tokens[j],tokens[j+1])==candidate:
                    new_tokens.append(tokens[j]+tokens[j+1])
                    j+=2
                else:
                    new_tokens.append(tokens[j])
                    j+=1
                    if j==n-1:
                        new_tokens.append(tokens[j])
            tokens=new_tokens
            results.append([candidate[0],candidate[1]])
        return results
            
     


