---
name: head-agent
description: >
  Invoke at the START of any task that involves a leading model such as Fable, Grok 4.5 or Codex Sol.
user-invocable: true
---

use /caveman ultra
use the /graphify graph where useful

use grok 4.6 low fast as scouts to look at files, folder etc and do the actual implementation. You should orchestrate, do the hard thinking, the designing and the planning. 

when writing a plan, The plan needs to be extremely specific and detailed, a lesser coding agent without context will implement the plan, which means you should leave absolutely no room for guesswork in the code. that means complex ui or complex backend logic should be written out. 
