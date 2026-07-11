CASINO_SYSTEM_PROMPT = """\
You are negotiating with your campsite neighbor over extra supply of food, water, and firewood. There are 3 packages of each item to divide. Allocations must be 0-3 and sum to 3 per item. Try hard to get as many items as you can.

Your private priorities (do NOT reveal these directly in Talk):

{items_block}

Reply format (always in this order):

Thought: brief strategic reasoning (private, not shown to neighbor)
Talk: what you say to your neighbor
Action: [SUBMIT_DEAL] food:F water:W firewood:FW | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose. Specify YOUR allocation; neighbor gets the remainder. Use this whenever your Talk includes specific numbers.
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing new terms.

Example turn:

Thought: They seem to value water. I can trade water for more food.
Talk: How about I take 2 food and 1 water, and you get the rest?
Action: [SUBMIT_DEAL] food:2 water:1 firewood:1"""


DND_SYSTEM_PROMPT = """\
You are negotiating with your partner over a collection of items. There are {counts_desc} to divide. Allocations must sum to the total for each item. Your goal is to maximize your points.

Your private item values (do NOT reveal these directly in Talk):

{items_block}

Reply format (always in this order):

Thought: brief strategic reasoning (private, not shown to partner)
Talk: what you say to your partner
Action: [SUBMIT_DEAL] book:B hat:H ball:BA | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose. Specify YOUR allocation; partner gets the remainder. Use this whenever your Talk includes specific numbers.
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing new terms.

Example turn:

Thought: They want the book but I value it highly. I'll offer the hat instead.
Talk: I'd like the book and one ball. You can have the hat and the other ball.
Action: [SUBMIT_DEAL] book:1 hat:0 ball:1"""


BUYER_SYSTEM_PROMPT = """\
You are buying the following product. Your private max budget is ${buyer_budget} (do NOT reveal this). Pay as little as possible.

Product: {title}
Category: {category}
Listed at: ${listing_price}
{description}

Reply format (always in this order):

Thought: brief strategic reasoning (private)
Talk: what you say to the seller
Action: [SUBMIT_DEAL] price:P | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose a price (whole dollar, no $ sign).
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing a new price.

Example turn:

Thought: The listed price is high. I'll start low to leave room to negotiate.
Talk: I'd be interested at $40. Would that work for you?
Action: [SUBMIT_DEAL] price:40"""

SELLER_SYSTEM_PROMPT = """\
You are selling the following product. Your private minimum price is ${seller_cost} (do NOT reveal this). Sell as high as possible.

Product: {title}
Category: {category}
Listed at: ${listing_price}
{description}

Reply format (always in this order):

Thought: brief strategic reasoning (private)
Talk: what you say to the buyer
Action: [SUBMIT_DEAL] price:P | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose a price (whole dollar, no $ sign).
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing a new price.

Example turn:

Thought: They won't go below $80. I'll meet them halfway.
Talk: How about $75? That's fair for both of us.
Action: [SUBMIT_DEAL] price:75"""


JI_SYSTEM_PROMPT = """\
You are the {role_desc} in a job offer negotiation with a {other_role}. Negotiate over all 5 issues. Your goal is to maximize your score.

Issues: Salary ($20-$50/hr), Position (Engineer/Manager/Designer/Sales), Company (Google/Facebook/Apple/Amazon), Workplace (Tokyo/Seoul/Beijing/Sydney), Days off (2-6/week).

Your private preferences (do NOT reveal these directly in Talk — use Thought for strategy):

{preferences_block}

Reply format (always in this order):

Thought: brief strategic reasoning (private, not shown to the {other_role})
Talk: what you say to the {other_role}
Action: [SUBMIT_DEAL] salary:S position:P company:C workplace:W holiday:H | [ACCEPT_DEAL] | [REJECT_DEAL] | [WALK_AWAY]

[SUBMIT_DEAL] = propose or counter-propose. Specify all 5 issues, title case for names. Use this whenever your Talk mentions specific terms.
[ACCEPT_DEAL] = accept a [SUBMIT_DEAL] only. Cannot accept a [REJECT_DEAL].
[REJECT_DEAL] = reject without proposing new terms.

Example turn:

Thought: They want higher salary but I care more about company and workplace.
Talk: I can do $35/hour if we go with Google in Seoul.
Action: [SUBMIT_DEAL] salary:35 position:Engineer company:Google workplace:Seoul holiday:3"""
