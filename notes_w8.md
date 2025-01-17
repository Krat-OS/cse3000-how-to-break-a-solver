# Agenda Meeting 8

## Details
### Location: 4.E450 De Bruijn, Building 28 (VMB)
### Date: 16th Jan 2025
### Time: 10:00-11:00 XOR 11:00-12:00
### Attendees: Anna, Cristian, Bram, David (chair), Vuk (note-taker)
## Agenda Items
### 1. No introduction due to time constrains

### 2. Update on Responsible Research (3 min)
- Coaching on Wednesday 15th
- No issues with GPDR/sensitive data
- Focus on integrity (respecting open-source licenses), reproducibility
- Acknowledge decisions made, limitations, use of GenAI
- Full notes in "Team" mattermost channel

Mention use of AI tools, acknowlegde use of computational resources.

Model counting doesn't really have any technical concerns compared to other fileds. There's a big discussion whether you should have an ethical comity at a CS conference, However, model counting is not affected. If one of application is verification of neural networks, there you could argue, but applications are not really our part. There's materials on responsible research.

### 3. Progress and questions round table (10 min / person)
- **Only what is relevant for Anna**
- Progress update
- Questions for Anna

#### Cristian:
Anna wanted to talk only about writing, however cristian will show some findings and. Anna encourages sharing and reading papers between each other.

Resilts section:
There's a table with instance types. There was a crash found, however, it was most likely due to the delftblue multithreading. He found a way to score the similarity. Every 2 instances are compared how similar they are in terms of results on the solver. Each instance of each geenrator is taken by agregating results. Appendix is allowed but not graded so he decided.
Anna: Can you show me the figure and tell me the biggest takeaway? From this you can conclude that 2 generators are more similar. Cristian for different generator uses different configurations of FuzzSAT (Brummayer). Were the results anticipated? Cristian believes they were but there were some unexpected. Do you think a methodlogy about selecting amount of data is sound? Anna is not sure, can't fully follow his thinking. Cristian decided to cut a chapter (Chapter about strategy choosing of instances with random forest), due to confusion caused. He computed difference between features and computed the contribution of that difference to the difficulty of instances.  He did tests 2 test if 2 groups are similar or not. The he calculated 2 features that have the most effect size. Features found were: CG-coeff-variation and VCG-CLAUSE_coeff-variation to be the most effective. He also wanted to explore miscounts but he didn't find any. Paper could make a strong tool for when the miscount is actually found. 

#### Bram:
Programming is done. Named EXTREMEgen. Experiments are currently being ran. there are no interesting results yet. He is considering changing the parameters for making it interestin. **There's no bad experiment. If you don't find anything interesting. Learn from it. That's intersting.** Bram found some more data from Texas rule. Do i need anything else then a table explaining bugs. Bram is worried he does not have enough box plots. He is looking into metamorphic testing. One thing you could do is if you find problems with. Going back to question what can a figure be made off, it depends on what you find. Look at your experiments, and if you think something could be communicated quicker by a figure then a table, use that. Can I get extension to the end of today to deliver a better draft? Tes! Writing today for bram, experiments tomorrow. Anna: feel free if you have that table to send it to me. And if you want to cite it use it as private communication:


#### David:
Improved existing sections
Added resulyts section
Added Responsible section
Added conclusion section
Considering improving in a number of tests, because CNFDD does a lot of tests and David does a lot less. What is a goog enough sample size. Don't overdo it 20 in 100000 is plenty. What originally said is fine. We canot really defend doinga  lot of instances. It's still a preliminary experiment. He generated a lot of instances and didn't find anything. Plan is to keep using timeouts and debug that. Text in intial paper draft v1 was bad, anna noted that, David was aware. Implemented a second heuristic and a third one is yet to be done. He didn't maange to put it in draft v2. He used a 15 minute timeout Have you already ran experiments on 100k instances. Vuk recommends looking into the script on mattermost for exclusively fuzzing. Also for spell checker, not the end of the world. It's a bachelor project. English is most people second language so Anna doesn't want to gatekeep science. You can cite a webpage if you want. 

Next meeting is on 23rd, next thirsyday, Bram is chair, 

Update: max length of paper is 8 pages excluding references and title.

### 4. Draft v2 feedback (3 min)
- When to expect it?
- Final paper deadline is 26th Jan, next meeting is 23rd Jan

### 5. General Questions & Feedback round (5 min)
- Any questions left unaddressed
- Any other feedback 
- Other matters of concern/needing improvement

### 6. Closing remarks (1 min)
- Goodbye everyone, see you next week Thursday!


To DO:

Anna needs to make sure readme in her work is correct so that Bram can cite it.