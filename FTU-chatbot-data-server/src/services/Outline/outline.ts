import { data } from '@/utils/data';
/* eslint-disable no-param-reassign */
/* eslint-disable no-underscore-dangle */
/* eslint-disable @typescript-eslint/naming-convention */
import axios from '@/utils/axios';
import { ip } from '@/utils/ip';


export interface IPayload {
  page: number,
  limit: number,
  cond: object,
}

class Topic<T> {
  name: string;
  url: string;

  constructor({ name, url }: { name: string; url: string }) {
    this.name = name;
    this.url = url || name;
  }

  del = async (id: string) => {
    return axios.delete(`${ip}/${this.url}/${id}`);
  };

  get = async (payload: any) => {
    // console.log(payload);
    return axios.get(`${ip}/api/topics`, { params: { populate: "*", ...payload } });
  };
  // get = async (payload: IPayload) => {
  //   // console.log(payload);
  //   return axios.get(`${ip}/${this.url}`, { params: payload });
  // };

  get_by_id = async (id: string) => {
    // console.log(payload);
    return axios.get(`${ip}/${this.url}`, { params: { intent_id: id } });
  };

  add = async (payload: T) => {
    return axios.post(`${ip}/${this.url}`, payload);
  };

  add2 = async (payload: T) => {
    Object.keys(payload).forEach((key) => {
      // payload[key] = payload[key];
      payload[key] = payload[key];
    });
    return axios.post(`${ip}/${this.url}`, payload);
  };

  upd = async (payload: T & { id: string | undefined, data: any }) => {
    const { id } = payload;
    payload.id = undefined;
    return axios.put(`${ip}/${this.url}/${id}`, payload.data);
  };

}

const topic = new Topic({ name: 'Topic', url: 'api/topics' });

export default topic;
