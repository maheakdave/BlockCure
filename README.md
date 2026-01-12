# BlockCure

Implemention of a blockchain based healthcare marketplace with Agentic AI health assistant

Paper reference -> [BlockCure](https://www.taylorfrancis.com/chapters/edit/10.1201/9781003428459-9/blockcure%E2%80%94an-anonymized-patient-data-sharing-platform-using-consortium-blockchain-shivesh-krishna-mukherjee-maheak-dave-rituparna-bhattacharya-jayanta-poray)

Folder Structure is as: 

SRC
  - FRONTED
      - src
           - components
           - Hooks
           - types
  - DB
  - BACKEND
      - SERVICES
      - UTILS

TODO:

- [x] Backend setup.
- [x] Database Setup.
- [ ] Completing UI.
- [ ] Integration of layers.
- [ ] Adding Queue-Worker System using Celery.
- [ ] Using triton inference server for deidentifier-api.
- [ ] Integrating Consensus mechanism for the blockchain structure.


TO setup project locally simply run:
```
cd SRC
docker compose up -d --build
```
