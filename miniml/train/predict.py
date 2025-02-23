from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


inputs = [
    "Great breakfast, good price. You might have to stand outside in line though, so I don't really recommend winter time to go. lol. Very friendly service, interesting coffee mugs. They have great deserts and such also. Bring your cash though as they dont' take cards.",
    "Been going to Dr. Goldberg for over 10 years. I think I was one of his 1st patients when he started at MHMG. He's been great over the years and is really all about the big picture. It is because of him, not my now former gyn Dr. Markoff, that I found out I have fibroids. He explores all options with you and is very patient and understanding. He doesn't judge and asks all the right questions. Very thorough and wants to be kept in the loop on every aspect of your medical health and your life.",
    """THE WORST DENTAL EXPERIENCE OF MY LIFE. THEY ARE BUTCHERS! My husband and I went to Emmert Dental in Bethel Park for a routine cleaning and check up. I had one tooth that I had a recent root canal on and a few small cavities that needed filled. After the so called dentist - Dr Carnavali - did my exam, I needed 3 additional root canals and an extraction! An extraction on the tooth that just had a root canal. I was told that part of the tooth had broken off and that a crown was no longer possible. I didn\'t know any better, so I agreed to schedule the extraction for a few weeks later. The "dentist" wanted to extract it that day, but I had plans for the weekend and was not about to be in pain. So, 2 weeks later, I went in for the extraction. After they gave me novocaine they get me with "you need a bone graft also in that area since we are extracting that tooth". I have never had any issues with bone loss with previous extractions so I didn\'t feel it was necessary. I had 3 employees surrounding me, pressuring me for this bone graft. So, I reluctantly agreed. I had to pay $259.00 for this "bone graft". So, I signed everything. I wasn\'t even given enough time to read the document, so I should have known and got up and left right then and there. The "dentist" started the extraction. I have never been so uncomfortable in my life. He pulled on my tooth, used a metal instrument to pound on my tooth and was pulling so hard that my head was coming up and slamming back down against the head rest. Finally, after a good 15 minutes of this torture, he ripped the tooth out of the side of my jaw. I was literally ganging from all the blood running down the back of my throat. He then stuck some pink powder up in the extraction area. I\'m assuming this was the bone graft. I ended up with many stitches. I was given an antibiotic and instructed to schedule an appointment for the following Saturday so they could start the bridge. Then I was hit with a bill of $569.00! My insurance covers 100% of extractions. I was in so much pain for 2 weeks following the extraction. Pain severe enough that I had to call off work and I couldn\'t sleep at night. I refused to go back to Emmert Dental. And after a visit to a very reputable dentist, an endodontist and an oral surgeon, I have to have oral surgery to try and repair the damage Emmert Dental has done to my mouth. They caused a very large deformity in my jaw and I am at risk to lose additional teeth. Save yourself a lot of time, money and pain. Emmert dental only cares about the money, will over charge you and leave you less than happy with the dental work. They are butchers and should not even call themselves dentist. And if you have an issue with their billing practices, don\'t expect to that resolved easily. I have called numerous times and I am always told that the manager and or billing department is not in. I have contacted the BBB and a lawyer. I expect and want a full refund. No one should have to go thru what I am currently going thru. They do unnecessary extractions so they can do a more expensive procedure such as a bridge or an implant. Now I have to go thru oral surgery, more pain and suffering and more time missed from work because of Emmert Dental. And the root canals they said I needed, my current dentist and my endodontist both agreed that I do not need tooth canals. In fact, the teeth are perfectly healthy. They are crooks and butchers""",
]

tokenizer = AutoTokenizer.from_pretrained("./shared/models/saved_model")
model = AutoModelForSequenceClassification.from_pretrained(
    "./shared/models/saved_model"
)

predict_pipeline = pipeline(
    "text-classification",
    model=model,
    device="cpu",
    tokenizer=tokenizer,
    truncation=True,
    max_length=512,
)

for review in inputs:
    prediction = predict_pipeline(review)
    print(prediction)
