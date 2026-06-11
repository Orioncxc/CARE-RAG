# Consequence Taxonomy Expansion Sources

This policy-store expansion uses public safety benchmark taxonomies to identify
coverage gaps. Benchmark prompts are not embedded in the consequence store.
Entries remain short reviewed policies linked to authoritative consequence
sources.

## Public Benchmark Coverage Basis

| Dataset / Benchmark | Published Coverage Used For Gap Analysis | Added Policy Domains |
| --- | --- | --- |
| BeaverTails | 14 categories including child abuse, discrimination, drug abuse, self-harm, terrorism/organized crime, and violence | `child_exploitation_safety`, `hate_safety`, `controlled_substance_safety`, `self_harm_safety`, `terrorism_safety` |
| Aegis 2.0 | Risk taxonomy including suicide/self-harm, sexual-minor, controlled substances, criminal planning, threats, and privacy | `self_harm_safety`, `child_exploitation_safety`, `controlled_substance_safety`, `sexual_violence_safety` |
| SALAD-Bench | Hierarchical categories including self-harm, privacy infringement, dangerous information, cyberattack, biological/chemical harms, and attacks on critical infrastructure | `self_harm_safety`, `critical_infrastructure_safety` |
| AIR-Bench 2024 | Regulation- and policy-grounded taxonomy with 314 fine-grained risk categories | Supports the use of authority-linked, auditable consequence policies rather than benchmark-only category names |
| WildGuardMix | In-the-wild moderation data spanning abuse, violence, self-harm, sexual content, misinformation, and discrimination | Supports separate protective-vs-facilitative handling for broad content safety risks |

## Added High-Consequence Domains

| Domain | Harmful Facilitation Policy | Protective Path | Authority Metadata Source |
| --- | --- | --- | --- |
| `self_harm_safety` | Suicide or self-injury encouragement/assistance | Crisis support and prevention | SAMHSA 988 |
| `child_exploitation_safety` | Child sexual exploitation or grooming | Safeguarding and reporting | U.S. DOJ |
| `hate_safety` | Bias-motivated violence or threats | Victim support and reporting | U.S. DOJ Hate Crimes |
| `controlled_substance_safety` | Illegal production/distribution/diversion | Overdose prevention and reporting | DEA |
| `terrorism_safety` | Terrorist or violent-extremist facilitation | Threat reporting and prevention | FBI |
| `critical_infrastructure_safety` | Sabotage of essential services | Resilience and incident reporting | CISA |
| `sexual_violence_safety` | Sexual violence or coercive exploitation | Survivor support and reporting | U.S. DOJ OVW |

## External Sources

- BeaverTails repository and category list: <https://github.com/PKU-Alignment/beavertails>
- Aegis 2.0 paper: <https://arxiv.org/abs/2501.09004>
- SALAD-Bench paper and repository: <https://arxiv.org/abs/2402.05044>, <https://github.com/OpenSafetyLab/SALAD-BENCH>
- AIR-Bench 2024 repository: <https://github.com/stanford-crfm/air-bench-2024>
- WildGuardMix dataset card: <https://huggingface.co/datasets/allenai/wildguardmix>
- SAMHSA 988: <https://www.samhsa.gov/find-help/988>
- U.S. DOJ Child Exploitation Prevention: <https://www.justice.gov/psc>
- U.S. DOJ Hate Crimes: <https://www.justice.gov/hatecrimes>
- DEA Drug Information: <https://www.dea.gov/drug-information>
- FBI Terrorism: <https://www.fbi.gov/investigate/terrorism>
- CISA Critical Infrastructure Sectors: <https://www.cisa.gov/topics/critical-infrastructure-security-and-resilience/critical-infrastructure-sectors>
- U.S. DOJ Office on Violence Against Women: <https://www.justice.gov/ovw>
