
######################################
#   MultiNLI Annotation Tags Script  #
#   By Adina Williams                #
######################################

# How to use this script:
# You should have a version of MultiNLI downloaded
# You should update the paths to point to your local files
# This can take MultiNLI dev set, test set, matched and mismatched, as well as SNLI files


tags_to_results = defaultdict(list)

def log(tag, is_correct, label):
    tags_to_results[tag].append((is_correct, label))

def find_1st_verb(str1):  #find ptb verb codings for first verb from root of sentence
    findy=str1.find('(VB')
    if findy >0:
        return str1[findy:].split()[0]
    else: 
        return ''

def tense_match(str1,str2):   # this function test for tense match, by finding the first verb and checking it against the second verb occurence
    result=find_1st_verb(str1)
    if len(result)>0:
        findy2=str2.find(result)
        return findy2>0
    else:
        return False

##################
#   declare      #
#   your paths   #
##################

dev_m_path = '/Users/Adina/Documents/MultiGenreNLI/multinli/multinli_1.0_dev_matched.txt'
dev_mm_path = '/Users/Adina/Documents/MultiGenreNLI/multinli/multinli_1.0_dev_mismatched.txt'
test_m_path = '/Users/Adina/Documents/MultiGenreNLI/multinli/multinli_1.0_test_matched.txt'
test_mm_path = '/Users/Adina/Documents/MultiGenreNLI/multinli/multinli_1.0_test_mismatched.txt'
dev_path = '/Users/Adina/Documents/MultiGenreNLI/multinli/multinli_1.0_dev_all.txt' #  concatenated match and mismatch
test_path = '/Users/Adina/Documents/MultiGenreNLI/mnli_0.9/multinli_0.9_test_all.txt' #  concatenated match and mismatch
snli_test_path = '/Users/Adina/Documents/MultiGenreNLI/snli_1.0/snli_1.0_test.txt'
snli_dev_path = '/Users/Adina/Documents/MultiGenreNLI/snli_1.0/snli_1.0_dev.txt'
#train_path = '/Users/Adina/Documents/MultiGenreNLI/mnli_0.9/multinli_0.9_train.txt'

ptbtags={"(MD":"modal","(W":"WH","(CD":"card","(PRP":"pron","(EX":"exist","(IN":"prep","(POS":"'s"} # dict of interesting ptb tags to check, pulled directly from PTB tagger

# for the dataset you want to see annotations for. You should specify its path, above. 
#  this gets your data in  a reasonable order for figuring out the annotations
#  this part also selects the PTB parses from the corpus. It works with the .txt distribution but not the jsonl distribution

with open(dev_path, 'rbU')as csvfile:  
    reader = csv.reader(csvfile, delimiter="\t")
    i = 0
    for row in reader:
        label = row[0]
        if label in ["entailment", "contradiction", "neutral"]:
            pairid = row[8]
            p1 = row[3]
            p2 = row[4]
            b1 = row[1]
            b2 = row[2]
            t1 = row[5]
            t2 = row[6]
            genre = row[9]
            b1t = b1.split()
            b2t = b2.split()
            sb1t = set(b1t)
            sb2t = set(b2t)
            parses = p1 + " " + p2
            correct = 'correct' # this needs to be supplied from the model outputs

            log("label-" + label, correct, label)


####################
# HAND CHOSEN TAGS #
####################
# Linguistically   #
# interesting      #
# phenomena were   #
# picked by Adina  #
####################

# just a note, results will be reported for both hypotheses and pairs (i.e., hypothesis or premise)


################
#   NEG/DET    #
################

            if "n't" in parses or "not" in parses or "none" in parses or "never" in parses or "neither" in parses or "nor" in parses:  # add in un- and non- :/
                log('neg-all', correct, label)
                if ("n't" in p2 or "not" in p2 or "none" in p2 or "never" in p2 or "neither" in p2 or "nor" in p2) and not ("n't" in p1 or "not" in p1 or "none" in p1 or "never" in p1 or "neither" in p1 or "nor" in p1):
                    log('neg-hyp-only', correct, label)


            if "a" in parses or "the" in parses or "these" in parses or "this" in parses or "those" in parses or "that" in parses: 
                log('det-all', correct, label)
                if ("a" in p2 or "the" in p2 or "these" in p2 or "this" in p2 or "those" in p2 or "that" in p2) and not ("a" in p1 or "the" in p1 or "these" in p1 or "this" in p1 or "those" in p1 or "that" in p1):
                    log('det-hyp-only', correct, label)

##################
#    PTB TAGS    #
##################
            for key in ptbtags:
                if key in parses:
                    log(ptbtags[key]+'_ptb_all', correct, label)
                if (key in p2) and not (key in p1):
                    log(ptbtags[key]+'_ptb_hyp_only', correct, label)

            if ("(NNS"  in p2) and ("(NNP" in p1):
                log('plural-premise-sing-hyp_ptb', correct, label)
            if ("(NNP"  in p2) and ("(NNS" in p1):
                log('plural-hyp-sing-premise_ptb', correct, label)

            if tense_match(p1,p2):
                log('tense_match', correct, label)
###################
#  interjects &   #  # we added some extra, potentially interesting things to check, but they didn't turn out to be interesting.
#  foreign words  #
###################

            if "(UH" in parses: 
                log('interject-all_ptb', correct, label)
                if ("(UH"  in p2) and not ("(UH" in p1):
                    log('interject-hyp-only_ptb', correct, label)

            if "(FW" in parses: 
                log('foreign-all_ptb', correct, label)
                if ("(FW"  in p2) and not ("(FW" in p1):
                    log('foreign-hyp-only_ptb', correct, label)

###################
#  PTB modifiers  #
###################

            if "(JJ" in parses: 
                log('adject-all_ptb', correct, label)
                if ("(JJ"  in p2) and not ("(JJ" in p1):
                    log('adject-hyp-only_ptb', correct, label)

            if "(RB" in parses: 
                log('adverb-all_ptb', correct, label)
                if ("(RB"  in p2) and not ("(RB" in p1):
                    log('adverb-hyp-only_ptb', correct, label)

            if "(JJ" in parses or "(RB" in parses: 
                log('adj/adv-all_ptb', correct, label)
                if ("(JJ"  in p2 or "(RB" in p2) and not ("(JJ" in p1 or "(RB" in p1):
                    log('adj/adv-hyp-only_ptb', correct, label)
# modifiers are good examples of how additions/subtractions of single words result in neutral

# if hyp (and premise) have -er -est adjectives or adverbs in them
            if "(RBR" in parses or "(RBS" in parses or "(JJR" in parses or "(JJS" in parses: 
                log('er-est-all_ptb', correct, label)
                if ("(RBR"  in p2 or "(RBS" in p2 or "(JJR" in p2 or "(JJS" in p2) and not ("(RBR" in p1 or "(RBS" in p1 or "(JJR" in p1 or "(JJS" in p1):
                    log('er-est-hyp-only_ptb', correct, label)




#########################
#  S-Root, length etc.  #
#########################

            s1 = p1[0:8] == "(ROOT (S"
            s2 = p2[0:8] == "(ROOT (S" 
            if s1 and s2:
                log('syn-S-S', correct, label)
            elif s1 or s2:
                log('syn-S-NP', correct, label)
            else:
                log('syn-NP-NP', correct, label)

            prem_len = len([word for word in b2.split() if word != '(' and word != ')'])
            if prem_len < 11:
                log('len-0-10', correct, label)
            elif prem_len < 15:
                log('len-11-14', correct, label)
            elif prem_len < 20:
                log('len-15-19', correct, label)
            else:
                log('len-20+', correct, label)

            if sb1t.issubset(sb2t):
                log('token-ins-only', correct, label)
            elif sb2t.issubset(sb1t):
                log('token-del-only', correct, label)


            if len(sb1t.difference(sb2t)) == 1 and len(sb2t.difference(sb1t)) == 1:
                log('token-single-sub-or-move', correct, label)

            if len(sb1t.union(sb2t)) > 0:
                overlap = float(len(sb1t.intersection(sb2t)))/len(sb1t.union(sb2t)) 
                if overlap > 0.6:
                    log('overlap-xhigh', correct, label)
                elif overlap > 0.37:
                    log('overlap-high', correct, label)
                elif overlap > 0.23:
                    log('overlap-mid', correct, label)
                elif overlap > 0.12:
                    log('overlap-low', correct, label)
                else:
                    log('overlap-xlow', correct, label)
            else: 
                log('overlap-empty', correct, label) 


##############
#   GREPing  #
##############



#            for keyphrase in ["there are", "there is", "There are", "There is", "There's", "there's", "there were", "There were", "There was", "there was", "there will", "There will"]:
#                if keyphrase in t2:
#                   log('template-thereis', correct, label)
#                   break

#            for keyphrase in ["can", "could", "may", "might", "must", "will", "would", "should"]:
#                if keyphrase in p2 or keyphrase in p1:
#                   log('template-modals', correct, label)
#                   break

            for keyphrase in ["much", "enough", "more", "most", "every", "each", "less", "least", "no", "none", "some", "all", "any", "many", "few", "several"]:  # get a list from Anna's book, think more about it
                if keyphrase in p2 or keyphrase in p1:
                    log('template-quantifiers', correct, label)
                    break

            for keyphrase in ["know", "knew", "believe", "understood", "understand", "doubt", "notice", "contemplate", "consider", "wonder", "thought", "think", "suspect", "suppose", "recognize",  "recognise", "forgot", "forget", "remember",  "imagine", "meant", "agree", "mean",  "disagree", "denied", "deny", "promise"]:
                if keyphrase in p2 or keyphrase in p1:
                    log('template-beliefVs', correct, label)
                    break

#           for keyphrase in ["love", "hate", "dislike", "annoy", "angry",  "happy", "sad", "bliss", "blissful", "depress","terrified","terrify", "scare", "amuse", "suprise", "guilt", "fear", "afraid", "startle",  "confuse", "baffle", "frustrate", "enfuriate", "rage", "befuddle", "fury", "furious", "elated", "elation", "joy", "joyous", "joyful", "enjoy", "relish"]:
#               if keyphrase in p2 or keyphrase in p1:
#                   log('template-psychpreds', correct, label)
#                   break


#           for keyphrase in ['I', 'me', 'my', 'mine', 'we', 'our', 'ours', 'you', 'your', 'yours', "y'all", 'he', 'him', 'her', 'she', 'it', 'they', 'their', 'theirs', 'them']:#
#               if keyphrase in t2:
#                   log('template-pronouns', correct, label)
#                   break

            for keyphrase in ['if']:
                    if keyphrase in p2 or keyphrase in p1:
                        log('template-if', correct, label)
                        break

#           for keyphrase in ["May I", "Mr.", "Mrs." "Ms.", "Dr.", "excuse me", "Excuse me", "pardon me", "sorry", "Sorry", "I'm sorry", "I am sorry", "Pardon me", 'please', 'thank', 'thanks', 'Thanks', 'Thank', 'Please', "you're welcome", "You're welcome", "much obliged", "Much obliged"]:
#                   if keyphrase in p2 or keyphrase in p1:
#                       log('template-polite', correct, label)
#                       break

            for keyphrase in ["time", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "morning", "night", "tomorrow", "yesterday", "evening", "week", "weeks", "hours", "minutes", "seconds" "hour", "days", "years", "decades", "lifetime", "lifetimes", "epoch", "epochs", "day", "recent", "recently", "habitually", "whenever", "during", "while", "before", "after", "previously", "again", "often", "repeatedly", "frequently", "dusk", "dawn", "midnight", "afternoon", "when", "daybreak", "later", "earlier", "month", "year", "decade", "biweekly", "millenium", "midday", "daily", "weekly", "monthly", "yearly", "hourly", "fortnight", "now", "then"]:
                    if keyphrase in p2 or keyphrase in p1:
                        log('template-timeterms', correct, label)
                        break

            for keyphrase in ["too", "anymore", "also", "as well", "again", "no longer", "start", "started", "starting", "stopping", "stop", "stopped", "regretting", "regret", "regretted", "realizing", "realize", "realized", "aware", "manage", "managed", "forgetting", "forget", "forgot", "began", "begin", "finish", "finished", "finishing", "ceasing", "cease", "ceased", "enter", "entered", "entering", "leaving", "leave", "left", "carry on", "carried on", "return", "returned", "returning", "restoring", "restore", "restored", "repeat", "repeated", "repeating", "another", "only", "coming back", "come back", "came back"]:
                    if keyphrase in p2 or keyphrase in p1:
                        log('template-presupptrigs', correct, label)
                        break

            for keyphrase in ["although", "but", "yet", "despite", "however", "However", "Although", "But", "Yet", "Despite", "therefore", "Therefore", "Thus", "thus"]:
                    if keyphrase in p2 or keyphrase in p1:
                        log('template-convo-pivot', correct, label)
                        break                          

#            for keyphrase in ["weight", "height", "age", "width", "length", "mother", "father", "sister", "brother", "aunt", "uncle", "cousin", "husband", "wife", "mom", "dad", "Mom", "Dad", "Mama", "Papa", "mama", "papa", "grandma", "grandpa", "nephew", "niece", "widow", "family", "kin", "bride", "spouse"]:
#                   if keyphrase in p2 or keyphrase in p1:
#                       log('template-relNs', correct, label)
#                       break      

#            for keyphrase in ['who', 'what', 'why', 'when', 'how', "where", "which", "whose", "whether"]:
#                    if keyphrase in t2:
#                        log('template-WH', correct, label)
#                        break

#            for keyphrase in ['for', 'with', 'in', 'of', 'on', 'at', 'into', 'by', 'through', 'via', 'throughout', 'near', 'up', 'down', 'off', 'over', 'under', 'underneath', 'against', 'above', 'to', 'towards', 'toward', 'until', 'away', 'from', 'beneath', 'beside', 'within', 'without', 'upon', 'onto', 'aside', 'across', 'about', 'after', 'before', 'along', 'among', 'around', 'after', 'between', 'beyond', 'below']:
#                    if keyphrase in t2:
#                        log('template-prep', correct, label)
#                        break
 
###############################
#  Too few to be interesting  #
###############################

#            if ("(RBR"  in p2  or "(JJR" in p2 ) and ("(RBS" in p1 or "(JJS" in p1):
#                log('er-hyp-est-premise_ptb', correct, label)
#            
#            if ("(RBS"  in p2  or "(JJS" in p2 ) and ("(RBR" in p1 or "(JJR" in p1):
#                log('est-hyp-er-premise_ptb', correct, label)
#
#            if "(PDT" in parses: 
#                log("pre-det-all_ptb", correct, label)
#                if ("(PDT"  in p2) and not ("(PDT" in p1):
#                    log("pre-det-hyp-only_ptb", correct, label)
#            for keyphrase in ["at home", "at school", "home", "to school", "from school", "at church", "from church", "to church", "in jail", "in prison"]:
#                    if keyphrase in t2:
#                        log('template-PPincorp', correct, label)
#                        break   
#            for keyphrase in ["more", "less", "than"]:
#                    if keyphrase in t2:
#                        log('template-more/less', correct, label)
#                        break  
#            for keyphrase in ['fake', 'false', 'counterfeit', 'alleged', 'former','mock', "imitation"]:
#                    if keyphrase in t2:
#                        log('template-nonsubsectAdj', correct, label)
#                        break


            i += 1

# This will print your results to the terminal and create a summary .csv file in current directory that saves them for you

with open('snli_autoannotationTags_results.csv', 'w') as csvfile:  #you name your file here
    writer = csv.writer(csvfile, delimiter='\t')
    for tag in sorted(tags_to_results):
        correct = len([result[0] for result in tags_to_results[tag] if result[0]])
        counts = Counter([result[1] for result in tags_to_results[tag]])
        best_label, best_count = max(counts.iteritems(), key=operator.itemgetter(1))

        attempted = len(tags_to_results[tag])
        baseline = float(best_count) / attempted 

        acc = float(correct)/attempted
        totalpercent= float(attempted)/i

        print tag, "\t", correct, "\t", attempted, "\t", acc, "\t", baseline, "\t", best_label, "\t", totalpercent
        writer.writerow([tag, correct, attempted, acc, baseline,  best_label, totalpercent])

