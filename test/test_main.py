import pytest
from src.main import llm, create_faq_agent, Chatbot
from utils import create_dataset_if_not_exists, evaluate

@pytest.fixture
def faq_chat_examples():
    return [
        # --- AUTO COVERAGE ---
        {
            "inputs": {"question": "What is liability car insurance?"},
            "outputs": {"answer": "Liability coverage pays another party's medical expenses, vehicle repairs, and property damage if you were responsible for the accident. It does not cover your own vehicle or your own injuries."},
        },
        {
            "inputs": {"question": "What is collision coverage?"},
            "outputs": {"answer": "Collision coverage helps pay to repair or replace your vehicle if it overturns or collides with another vehicle or object, such as a tree or guardrail."},
        },
        {
            "inputs": {"question": "What is comprehensive coverage?"},
            "outputs": {"answer": "Comprehensive coverage helps pay to repair or replace your vehicle if it's damaged by something other than a collision, including theft, fire, vandalism, or hitting an animal."},
        },
        {
            "inputs": {"question": "What is full coverage car insurance?"},
            "outputs": {"answer": "Full coverage is not a formally defined insurance term, but it typically refers to having both liability and physical damage coverage, including comprehensive and collision coverage."},
        },
        {
            "inputs": {"question": "What is Medical Payments coverage?"},
            "outputs": {"answer": "Medical Payments coverage helps pay medical and funeral expenses if an insured person, including a passenger, is injured or killed in an accident, regardless of who is at fault."},
        },
        {
            "inputs": {"question": "What is personal injury protection?"},
            "outputs": {"answer": "Personal injury protection, or PIP, covers medical expenses, lost wages, and other costs if you are injured in an accident, regardless of fault. PIP typically applies before your health insurance."},
        },
        {
            "inputs": {"question": "What is uninsured motorist coverage?"},
            "outputs": {"answer": "Uninsured motorist coverage helps pay for your medical expenses, pain and suffering, and lost wages when you are hit by a driver who has no insurance."},
        },
        {
            "inputs": {"question": "What does 100/300/100 mean in car insurance?"},
            "outputs": {"answer": "These numbers refer to liability coverage limits: up to $100,000 for bodily injury per person, up to $300,000 total for bodily injury per accident, and up to $100,000 for property damage per accident."},
        },
        {
            "inputs": {"question": "When should I choose liability-only versus full coverage?"},
            "outputs": {"answer": "If your car is older and paid off, liability-only coverage may be enough. If you have a newer vehicle, a loan or lease, or want added protection for your own car, adding comprehensive and collision coverage is typically the better choice."},
        },
        {
            "inputs": {"question": "Is liability car insurance required in all states?"},
            "outputs": {"answer": "Liability car insurance is required in most states. Only a few states, like New Hampshire and Virginia, allow drivers to operate vehicles without liability insurance, but they have alternative requirements such as proof of financial responsibility."},
        },
        # --- CLAIMS PROCESS ---
        {
            "inputs": {"question": "What is a third-party insurance claim?"},
            "outputs": {"answer": "A third-party insurance claim is a claim you file with someone else's insurance company after they cause an accident."},
        },
        {
            "inputs": {"question": "How do I submit an auto insurance claim?"},
            "outputs": {"answer": "You can file a claim by logging in to your policy online, through your insurer's mobile app, or by calling the claims center."},
        },
        {
            "inputs": {"question": "How is fault determined after an accident?"},
            "outputs": {"answer": "Liability, or fault, will be determined based on state laws and the circumstances of the accident. Depending on the facts of the loss, there may be shared responsibility between the parties involved."},
        },
        {
            "inputs": {"question": "Who pays my deductible if I'm not at fault?"},
            "outputs": {"answer": "Your insurer will work to recover your deductible from the at-fault party or their insurance company, though recovery isn't guaranteed. If there is shared responsibility, the amount you receive back may be prorated based on the percentage of fault."},
        },
        {
            "inputs": {"question": "Do I have to get my car repaired after a claim?"},
            "outputs": {"answer": "The decision to repair is yours — and it's okay to not repair your vehicle. However, if a lienholder exists on your vehicle, they may require repairs."},
        },
        {
            "inputs": {"question": "Do I have to report an accident even if I'm not at fault?"},
            "outputs": {"answer": "You are required to report a claim even if it's not your fault. Reporting protects your interests, especially when injuries or property damage have occurred."},
        },
        {
            "inputs": {"question": "How long do car repairs take after a claim?"},
            "outputs": {"answer": "Many property damage claims are resolved within 7 to 14 days, but repair times can vary greatly based on your vehicle, the extent of the damage, and parts availability."},
        },
        {
            "inputs": {"question": "Will filing a claim raise my premium?"},
            "outputs": {"answer": "Your claim will not impact your current rate. However, when you renew your policy, your rate may increase depending on the claim. Claims under $500 or claims that go unpaid typically don't raise rates."},
        },
        {
            "inputs": {"question": "What is an insurance adjuster?"},
            "outputs": {"answer": "An insurance adjuster, also known as a claims adjuster, is a person who investigates an insurance claim to determine if the insurer should pay for damage or injuries, and if so, how much they should pay."},
        },
        {
            "inputs": {"question": "What are my repair and inspection options after filing a claim?"},
            "outputs": {"answer": "Options typically include using a network repair shop recommended by your insurer, choosing any repair shop of your preference, or using a photo estimate tool through your insurer's mobile app to document damage remotely."},
        },
        # --- TOTAL LOSS ---
        {
            "inputs": {"question": "What does it mean if my vehicle is a total loss?"},
            "outputs": {"answer": "Generally, a vehicle is a total loss when the cost to return it to its pre-loss condition is greater than the value of the vehicle."},
        },
        {
            "inputs": {"question": "How much will I receive for my total loss vehicle?"},
            "outputs": {"answer": "Your insurer pays you the actual cash value of your vehicle — the market value based on factors such as its pre-loss condition, age, options, and mileage — minus any applicable deductible."},
        },
        {
            "inputs": {"question": "Will I still owe money on my loan after a total loss?"},
            "outputs": {"answer": "You may still owe money if the vehicle's actual cash value is less than your remaining loan or lease balance. Loan/Lease Payoff coverage, also known as gap coverage, can help cover the difference up to your policy limits."},
        },
        {
            "inputs": {"question": "Will my insurance automatically transfer to my replacement vehicle after a total loss?"},
            "outputs": {"answer": "No. You must log in, call, or contact your agent to remove the totaled vehicle from your policy and add your replacement vehicle."},
        },
        # --- SPECIFIC COVERAGE SCENARIOS ---
        {
            "inputs": {"question": "Does car insurance cover vandalism?"},
            "outputs": {"answer": "Yes, comprehensive coverage on your auto policy can cover vandalism to your car — minus any deductible — since intentional damage to your vehicle is out of your control."},
        },
        {
            "inputs": {"question": "Does car insurance cover water damage?"},
            "outputs": {"answer": "Comprehensive coverage can protect your vehicle against water damage caused by flooding, heavy rains, hail, and even tree branches blown down during a storm."},
        },
        {
            "inputs": {"question": "Does car insurance cover hail damage?"},
            "outputs": {"answer": "If your vehicle is damaged by hail, auto comprehensive coverage may help pay to repair or even replace your vehicle, minus your deductible."},
        },
        {
            "inputs": {"question": "Does car insurance cover tire damage?"},
            "outputs": {"answer": "Car insurance may help cover tire damage if it's caused by a covered event, such as an accident, vandalism, theft, or severe weather, and you have comprehensive or collision coverage."},
        },
        {
            "inputs": {"question": "Does liability insurance cover rental cars?"},
            "outputs": {"answer": "Liability insurance typically covers rental cars, but only for damages or injuries you cause to others while driving the rental. It does not cover damage to the rental vehicle itself."},
        },
        {
            "inputs": {"question": "What happens if my damages exceed my liability coverage limits?"},
            "outputs": {"answer": "If your damages exceed your liability coverage, you are personally responsible for paying the remaining amount out of pocket."},
        },
        # --- HOMEOWNERS ---
        {
            "inputs": {"question": "What does homeowners insurance cover?"},
            "outputs": {"answer": "Standard homeowners insurance covers damage to your home's structure from covered perils such as fire, windstorm, hail, and lightning. It also covers personal belongings, liability if someone is injured on your property, and additional living expenses if you need to temporarily live elsewhere while your home is being repaired."},
        },
        {
            "inputs": {"question": "Does homeowners insurance cover flood damage?"},
            "outputs": {"answer": "Standard homeowners insurance policies do not cover flood damage. Flood coverage is provided through a separate flood insurance policy available through the National Flood Insurance Program (NFIP) or private insurers."},
        },
        {
            "inputs": {"question": "Does homeowners insurance cover earthquake damage?"},
            "outputs": {"answer": "No. Earthquake damage is not covered by standard homeowners insurance. You must purchase a separate earthquake insurance policy or endorsement."},
        },
        {
            "inputs": {"question": "What kind of water damage does homeowners insurance cover?"},
            "outputs": {"answer": "Homeowners insurance covers sudden and accidental water damage. It typically does not cover water damage from external flooding, gradual damage, or unresolved maintenance issues."},
        },
        {
            "inputs": {"question": "Does homeowners insurance cover mold damage?"},
            "outputs": {"answer": "Standard homeowners insurance policies typically do not cover mold damage caused by preventable issues such as lack of maintenance or water leaks."},
        },
        {
            "inputs": {"question": "Is homeowners insurance required by law?"},
            "outputs": {"answer": "It varies by state whether the law requires you to carry homeowners insurance. However, your mortgage lender will typically require certain coverages and amounts in order to issue a loan."},
        },
        {
            "inputs": {"question": "Is homeowners insurance tax deductible?"},
            "outputs": {"answer": "In most cases, homeowners insurance premiums are not tax deductible for a primary residence."},
        },
        # --- BILLING & POLICY ---
        {
            "inputs": {"question": "Why did I receive a cancellation notice?"},
            "outputs": {"answer": "Cancellation notices are sent when your payment due date has passed and the amount due has not been paid. They indicate the final deadline to make a payment before your coverage lapses."},
        },
        {
            "inputs": {"question": "What is an SR-22?"},
            "outputs": {"answer": "An SR-22 is a form that proves your car insurance meets the minimum coverages required by law. It is also known as a Certificate of Financial Responsibility."},
        },
        {
            "inputs": {"question": "What is a personal umbrella insurance policy?"},
            "outputs": {"answer": "A personal umbrella policy is a type of liability insurance that provides additional coverage beyond the limits of your homeowners, auto, or other personal insurance policies."},
        },
    ]

@pytest.fixture
def adversarial_examples():
    return [
        # --- OUT-OF-SCOPE (agent must say "I don't know") ---
        {
            "inputs": {"question": "What is the capital of France?"},
            "outputs": {"answer": "I don't know"},
        },
        {
            "inputs": {"question": "Can you recommend the best insurance company?"},
            "outputs": {"answer": "I don't know"},
        },
        {
            "inputs": {"question": "Does my insurance cover damage from war or nuclear events?"},
            "outputs": {"answer": "I don't know"},
        },
        {
            "inputs": {"question": "Can I insure my cryptocurrency holdings?"},
            "outputs": {"answer": "I don't know"},
        },
        {
            "inputs": {"question": "What is the stock price of Progressive Insurance?"},
            "outputs": {"answer": "I don't know"},
        },

        # --- MULTI-HOP (requires combining 2+ FAQ entries) ---
        {
            "inputs": {"question": "If a hailstorm damages my car and I have comprehensive coverage, will I pay a deductible?"},
            "outputs": {"answer": "Yes. Comprehensive coverage pays for hail damage, but you will still owe your deductible before the insurer pays the rest."},
        },
        {
            "inputs": {"question": "If I'm hit by an uninsured driver and my car is totaled, what happens?"},
            "outputs": {"answer": "Uninsured motorist coverage can help pay your medical expenses and losses. If the car is a total loss, your insurer pays the actual cash value minus your deductible."},
        },
        {
            "inputs": {"question": "I have PIP coverage — do I still need to file a claim if the accident wasn't my fault?"},
            "outputs": {"answer": "Yes. You are required to report a claim even if it is not your fault. PIP covers your medical expenses regardless of fault."},
        },
        {
            "inputs": {"question": "If my vehicle is totaled and I still have a loan, will insurance cover the full loan amount?"},
            "outputs": {"answer": "Not necessarily. Insurance pays the actual cash value of the vehicle. If that is less than your loan balance, Loan/Lease Payoff (gap) coverage can help cover the difference up to policy limits."},
        },
        {
            "inputs": {"question": "Does homeowners insurance cover my car if it's stolen from my driveway?"},
            "outputs": {"answer": "No. Homeowners insurance does not cover vehicle theft. You would need comprehensive coverage on your auto policy for that."},
        },

        # --- PARAPHRASED (different wording, same meaning as FAQ) ---
        {
            "inputs": {"question": "If someone crashes into me and they have no insurance, what happens to my medical bills?"},
            "outputs": {"answer": "Uninsured motorist coverage helps pay for your medical expenses, pain and suffering, and lost wages when you are hit by a driver who has no insurance."},
        },
        {
            "inputs": {"question": "How does my insurer figure out who's to blame after a crash?"},
            "outputs": {"answer": "Liability is determined based on state laws and the circumstances of the accident. There may be shared responsibility between the parties involved."},
        },
        {
            "inputs": {"question": "What happens if fixing my car costs more than the car is worth?"},
            "outputs": {"answer": "Your vehicle is considered a total loss when the repair cost exceeds its pre-loss value. Your insurer will pay the actual cash value minus your deductible."},
        },
        {
            "inputs": {"question": "Will my rate go up if I make a claim?"},
            "outputs": {"answer": "Your current rate will not change. However, your premium may increase at renewal depending on the claim. Claims under $500 or unpaid claims typically do not raise rates."},
        },
        {
            "inputs": {"question": "Does my home policy pay if a guest slips and falls at my house?"},
            "outputs": {"answer": "Yes. Standard homeowners insurance includes liability coverage if someone is injured on your property."},
        },

        # --- COMPARATIVE / SUB-PART (requires multi-step reasoning) ---
        {
            "inputs": {"question": "What's the difference between comprehensive and collision coverage, and which one covers a flood?"},
            "outputs": {"answer": "Collision covers damage from hitting another vehicle or object. Comprehensive covers non-collision events like theft, fire, vandalism, and weather — including flooding. Flood damage is covered by comprehensive, not collision."},
        },
        {
            "inputs": {"question": "What's the difference between PIP and Medical Payments coverage?"},
            "outputs": {"answer": "Both cover medical expenses after an accident regardless of fault. PIP is broader — it also covers lost wages and other costs, and typically applies before your health insurance. Medical Payments covers medical and funeral expenses only."},
        },
        {
            "inputs": {"question": "If I only have liability coverage and I crash into a tree, am I covered?"},
            "outputs": {"answer": "No. Liability coverage only pays for damages you cause to others. Damage to your own vehicle from hitting a tree requires collision coverage."},
        },
        {
            "inputs": {"question": "What does 100/300/100 mean and is that enough coverage?"},
            "outputs": {"answer": "100/300/100 means $100,000 bodily injury per person, $300,000 total bodily injury per accident, and $100,000 property damage per accident. Whether it is enough depends on your assets and risk — higher limits provide more protection if damages exceed the limits."},
        },
        {
            "inputs": {"question": "Should I use my insurer's repair shop or can I pick my own?"},
            "outputs": {"answer": "You have options: use a network repair shop recommended by your insurer, choose any shop of your preference, or use a photo estimate tool through your insurer's mobile app to document damage remotely."},
        },
    ]


def test_evaluate_adversarial(adversarial_examples: list):
    dataset_name = "Insurance FAQ Complex dataset"
    create_dataset_if_not_exists(dataset_name, adversarial_examples)

    faq_agent = create_faq_agent(llm)
    chatbot = Chatbot(faq_agent)

    def target(inputs: dict) -> dict:
        response = chatbot.chat(inputs["question"])
        return {"answer": response}

    evaluations = evaluate(target, dataset_name, "insurance-faq-adversarial-v1")

    total, score_sum = 0, 0.0
    low_scores = []

    for evaluation in evaluations:
        for result in evaluation["evaluation_results"]["results"]:
            total += 1
            score = float(result.score or 0)
            score_sum += score
            if score < 0.5:
                low_scores.append(f"[LOW] {result.key} (score={score:.2f}): {result.comment}")

    avg_score = score_sum / total if total > 0 else 0.0

    print(f"\n=== Adversarial Evaluation Results ===")
    print(f"Total evaluations: {total} (20 questions × 3 metrics)")
    print(f"Avg score: {avg_score:.2f}")
    if low_scores:
        print("Low scores (<0.5):")
        for s in low_scores:
            print(f"  {s}")

    # Lower threshold — adversarial questions are intentionally hard
    assert avg_score >= 0.55, f"Avg score {avg_score:.2f} is below 0.55 adversarial threshold"
    
def test_evaluate_faq_agent(faq_chat_examples: list):
    dataset_name = "Insurance FAQ agent dataset"
    create_dataset_if_not_exists(dataset_name, faq_chat_examples)

    faq_agent = create_faq_agent(llm)
    chatbot = Chatbot(faq_agent)

    def target(inputs: dict) -> dict:
        response = chatbot.chat(inputs["question"])
        return {"answer": response}

    evaluations = evaluate(target, dataset_name, "insurance-faq-eval-v2")

    # Ratio scoring: sum all scores, divide by total evaluations
    total, score_sum = 0, 0.0
    low_scores = []

    for evaluation in evaluations:
        for result in evaluation["evaluation_results"]["results"]:
            total += 1
            score = float(result.score or 0)
            score_sum += score
            if score < 0.5:
                low_scores.append(f"[LOW] {result.key} (score={score:.2f}): {result.comment}")

    avg_score = score_sum / total if total > 0 else 0.0

    print(f"\n=== Evaluation Results ===")
    print(f"Total evaluations: {total} (40 questions × 3 metrics)")
    print(f"Avg score: {avg_score:.2f} ({score_sum:.1f} / {total})")
    if low_scores:
        print("Low scores (<0.5):")
        for s in low_scores:
            print(f"  {s}")

    assert avg_score >= 0.70, f"Avg score {avg_score:.2f} is below 0.70 threshold"