import { data } from '@/utils/data';
/* eslint-disable no-param-reassign */
/* eslint-disable no-underscore-dangle */
/* eslint-disable @typescript-eslint/naming-convention */
import axios from '@/utils/axios';
import { ip, ip3 } from '@/utils/ip';


export interface IPayload {
  page: number,
  limit: number,
  cond: object,
}

class Model<T> {
  name: string;
  url: string;

  constructor({ name, url }: { name: string; url: string }) {
    this.name = name;
    this.url = url || name;
  }

  del = async (id: string) => {
    return axios.delete(`${ip}/${this.url}`, { params: { id } });
  };

  get = async (payload: IPayload) => {
    // console.log(payload);
    return axios.get(`${ip}/${this.url}`, { params: payload });
  };

  get_by_id = async (id: string) => {
    // console.log(payload);
    return axios.get(`${ip}/${this.url}`, { params: { model_id: id } });
  };

  add = async (payload: T) => {
    return axios.post(`${ip}/${this.url}`, payload);
  };
  activate = async (payload: T) => {
    return axios.post(`${ip}/ModelRasaView`, payload);
  };

  autotrain = async (token: string) => {
    return axios.get(`${ip3}/re_ingest`, { params: { jwt: token } });
  }
  exportnludata = async (check: string) => {
    return axios.get(`${ip}/export`, { params: { check } });
  }

  add2 = async (payload: T) => {
    Object.keys(payload).forEach((key) => {
      // payload[key] = payload[key];
      payload[key] = payload[key];
    });
    return axios.post(`${ip}/${this.url}`, payload);
  };

  upd = async (payload: T & { id: string | undefined, data: FormData }) => {
    const { id } = payload;
    payload.id = undefined;
    return axios.put(`${ip}/${this.url}`, payload.data);
  };

}

const model = new Model({ name: 'Model', url: 'ModelRasaView' });

export default model;
