from index import Indexer
from index import WordToPointerEntry, PostingsList, Posting

indexer = Indexer(out_dict="dictionary.txt", out_postings="postings.txt")
indexer.load()
pl_american = indexer.get_posting_list("american")
print(pl_american)
pl_usa = indexer.get_posting_list("u.s.a.")
print(pl_usa)

text = """ALCAN UPS ALUMINIUM INGOT AND BILLET PRICES
  Alcan Aluminium Ltd. in Montreal said
  it increased yesterday its prices for unalloyed ingot and
  extrusion billet by two cents a lb, effective with shipments
  beginning May 1.
      The new price for unalloyed ingot is 64.5 cents a lb while
  the new price for extrusion billet is 72.5 cents a lb.
      "We feel very confident about raising our prices because we
  see demand over supply as being sustainable for some time,"
  said Ian Rugeroni, Alcan's president of metal sales and
  recycling - U.S.A.
      Rugeroni said sheet and can bookings for Alcan aluminium
  were up at a time when the company's total 1.1 mln tonne North
  American smelter system had less than a week's supply.
      "We're short and we're buying," he said.
      Rugeroni added that Alcan expects the International Primary
  Aluminum Institute to report a drop in total non-Socialist
  stocks in February and March. He estimated supply in the latter
  month will have fallen 100,000 to 150,000 tonnes, based in part
  on current low inventories of aluminium in Japan and on the
  London Metal Exchange."""

print(indexer.preprocess_text(text))